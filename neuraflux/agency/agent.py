import datetime as dt
import os
import numpy as np
from typing import Callable

import dill
import pandas as pd

from neuraflux.agency.config.config_module import ConfigModule
from neuraflux.agency.control.control_module import ControlModule
from neuraflux.agency.control.pareto import ParetoFrontier
from neuraflux.agency.data.data_module import DataModule
from neuraflux.agency.prediction.prediction_module import PredictionModule
from neuraflux.global_variables import CONTROL_KEY, OAT_KEY
from neuraflux.local_typing import UidType, IndexType
from neuraflux.agency.tariffs import TariffFactory
from neuraflux.agency.products import ProductFactory
from neuraflux.schemas.control import DiscreteControl
from neuraflux.weather_ref import WeatherRef
from neuraflux.assets.building import Building
from neuraflux.assets.electric_vehicle import ElectricVehicle
from neuraflux.assets.energy_storage import EnergyStorage
from neuraflux.agency.control.control_utils import softmax


class Agent:
    def __init__(
        self,
        uid: UidType,
        data_module: DataModule,
        control_module: ControlModule,
        prediction_module: PredictionModule,
        config_module: ConfigModule,
        index: IndexType,
        weather_ref: WeatherRef,
    ) -> None:
        # Update internal attributes
        self.uid = uid
        self.update_time_reference(index)
        self.update_modules(
            data_module=data_module,
            control_module=control_module,
            prediction_module=prediction_module,
            config_module=config_module,
            weather_ref=weather_ref,
        )
        self.scaler_fitted: bool = False

        # Initialize tariff structure from config
        tariff = TariffFactory.create(self.agent_config.tariff)
        self.data_module.assign_tariff_structure(uid=self.uid, tariff=tariff)

        # Initialize product from config
        product = ProductFactory.create(self.agent_config.product)
        self.data_module.assign_product(uid=self.uid, product=product)

    def update_modules(
        self,
        data_module: DataModule,
        control_module: ControlModule,
        prediction_module: PredictionModule,
        config_module: ConfigModule,
        weather_ref: WeatherRef,
    ) -> None:
        # Modules
        self.data_module = data_module
        self.control_module = control_module
        self.prediction_module = prediction_module
        self.config_module = config_module
        self.weather_ref = weather_ref

        # Agent config and related quantities
        self.agent_config = self.get_agent_config()
        self.rl_config = self.agent_config.control.reinforcement_learning
        self.cpm = self.agent_config.data.control_power_mapping

    def update_time_reference(self, index: IndexType) -> None:
        self.previous_index = None if not hasattr(self, "index") else self.index
        self.index = index

    def assign_to_asset(self, asset) -> None:
        self.asset = asset

    def assign_to_shadow_asset(self, shadow_asset) -> None:
        self.shadow_asset = shadow_asset

    # -----------------------------------------------------------------------
    # CONFIG
    # -----------------------------------------------------------------------
    def get_agent_config(self):
        return self.config_module.get_agent_config(self.uid)

    # -----------------------------------------------------------------------
    # DATA
    # -----------------------------------------------------------------------
    def asset_data_collection(self) -> None:
        # Store the current normal signal values
        signals_dict = {}
        for signal in self.agent_config.data.tracked_signals:
            signal_value = self.asset.get_signal(signal)
            # Store array-like values separately
            if isinstance(signal_value, (list, tuple, np.ndarray)):
                for i, value in enumerate(signal_value):
                    signals_dict[signal + "_" + str(i + 1)] = value
            else:
                signals_dict[signal] = signal_value
        self._push_asset_signal_data_to_db(signals_dict)

        # Define control keys since there are multiple, based on n_controls
        control_keys = [
            CONTROL_KEY + "_" + str(i + 1) for i in range(self.asset.config.n_controls)
        ]

        # Extract control values (list)
        if self.previous_index is not None:
            control_values = self.asset.get_signal(CONTROL_KEY)
            control_signals_dict = {k: v for k, v in zip(control_keys, control_values)}
            self._push_asset_signal_data_to_db(
                control_signals_dict, self.previous_index
            )

    def get_asset_signals(self) -> pd.DataFrame:
        return self.data_module.get_augmented_history(
            self.uid, self.cpm, self.weather_ref
        )

    def scale_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        scaled_data = self.data_module.scale_data(self.uid, data)
        return scaled_data

    def fit_scalers_on_data(self) -> None:
        if not self.scaler_fitted:
            asset_history = self.data_module.get_augmented_history(
                uid=self.uid,
                controls_power_mapping=self.cpm,
                weather_ref=self.weather_ref,
            )
            columns_to_scale = self.rl_config.state_signals
            self._fit_scaling_dict_on_data(asset_history, columns_to_scale)
            self.scaler_fitted = True

    def _push_asset_signal_data_to_db(
        self, signals_dict: dict[str, float | int | str], index=None
    ) -> None:
        index = self.index if index is None else index

        self.data_module.push_asset_signal_data(self.uid, index, signals_dict)

    def _fit_scaling_dict_on_data(
        self, df: pd.DataFrame, columns_to_scale: list[str]
    ) -> None:
        self.data_module.fit_scaling_dict_from_df(self.uid, df, columns_to_scale)

    # -----------------------------------------------------------------------
    # CONTROL
    # -----------------------------------------------------------------------
    def policy_evaluation(self, time_now: dt.datetime) -> None:
        if hasattr(self, "control_buffer") and len(self.control_buffer) > 0:
            n_trajectories = 1
        else:
            if self.control_module.is_rl_model_ready(self.uid):
                n_trajectories = self.agent_config.control.n_trajectory_samples
            else:
                n_trajectories = 1
        history_df = self.data_module.get_augmented_history(
            uid=self.uid,
            controls_power_mapping=self.cpm,
            weather_ref=self.weather_ref,
        )
        # Trajectories simulation
        trajectory_len = self.agent_config.control.trajectory_length
        possible_actions = [int(k) for k in self.cpm.keys()]
        n_controls = self.asset.config.n_controls
        for _ in range(n_trajectories):
            trajectory = self.sample_trajectory(
                history_df,
                trajectory_len,
                time_now,
                self.agent_boltzmann_policy_or_random,
            )

            # Keep only from t to t+trajectory_len
            trajectory = trajectory.loc[trajectory.index >= time_now, :]

            # Interrupt and print trajectory if there are NaNs
            if trajectory.iloc[:-1].isna().any().any():
                print("TRACEBACK DETAILS")
                print(trajectory)
                raise ValueError("Faulty trajectory: NaNs found in df.")

            # if trajectory.shape[0] != trajectory_len:
            #    print("TRACEBACK DETAILS")
            #    print(trajectory)
            #    raise ValueError(f"Wrong trajectory len: {trajectory.shape[0]} (expected {trajectory_len}).")

            # Save specific trajectory
            self.data_module.push_trajectory_data(self.uid, time_now, trajectory)

        if isinstance(self.asset, ElectricVehicle):
            for all_same_control in [0, 1, 2]:
                trajectory = self.sample_trajectory(
                    history_df,
                    trajectory_len,
                    time_now,
                    lambda df: [all_same_control],
                )

                # Keep only from t to t+trajectory_len
                trajectory = trajectory.loc[trajectory.index >= time_now, :]

                # Interrupt and print trajectory if there are NaNs
                if trajectory.iloc[:-1].isna().any().any():
                    print("TRACEBACK DETAILS")
                    print(trajectory)
                    raise ValueError("Faulty trajectory: NaNs found in df.")

                # if trajectory.shape[0] != trajectory_len:
                #    print("TRACEBACK DETAILS")
                #    print(trajectory)
                #    raise ValueError(f"Wrong trajectory len: {trajectory.shape[0]} (expected {trajectory_len}).")

                # Save specific trajectory
                self.data_module.push_trajectory_data(self.uid, time_now, trajectory)

    def agent_boltzmann_policy_or_random(self, df, eps: float = 0.05) -> list[int]:
        if self.control_module.is_rl_model_ready(self.uid):
        
            # Keep only signals required for last state
            df = df.iloc[-self.rl_config.history_length :]
            scaled_df = self.data_module.scale_data(self.uid, df)
            q_factors = self.control_module.get_raw_q_factors(
                uid=self.uid,
                scaled_data=scaled_df,
                rl_config=self.rl_config,
            )

            controls = []
            for i, q_tensor in enumerate(q_factors):
                # Mean of all objectives
                q_values = np.mean(q_tensor[-1, :, :], axis=0)
                # print(f"  q_tensor {i+1} in boltzmann: {q_tensor}")
                # print(f"  q_values {i+1} in boltzmann: {q_values}")
                # ArgMax
                # control_int = np.argmax(q_values)

                # Boltzmann policy selection
                # probabilities = softmax(q_values)
                # Apply softmax to compute the probabilities
                softmax_probabilities = softmax(q_values)

                # Ensure each probability is at least epsilon
                probabilities = np.maximum(softmax_probabilities, eps)

                # Normalize the probabilities to sum to 1
                probabilities /= np.sum(probabilities)
                # print(f"  probabilities in boltzmann: {probabilities}")
                control_int = np.random.choice(range(len(q_values)), p=probabilities)

                # Do not allow discharge for empty energy storage
                if isinstance(self.asset, EnergyStorage):
                    if control_int == 0 and self.asset.internal_energy <= 0.0:
                        control_int = 1
                
                # EV checks 
                if isinstance(self.asset, ElectricVehicle):
                    # Should never discharge except in DR
                    if df.loc[df.index[0], "dr_event"] == False and control_int == 0:
                        control_int = 1

                    # If below 40%, automatically stop discharge
                    if self.asset.internal_energy < 30 and control_int == 0:
                        control_int += 1

                controls.append(int(control_int))
            return controls
        else:
            n_controls = self.asset.config.n_controls
            possible_actions = [int(k) for k in self.cpm.keys()]
            return np.random.choice(possible_actions, size=n_controls)

    def get_controls(self) -> DiscreteControl | None:
        if self.control_module.is_rl_model_ready(self.uid):
            # Get asset signals dataset
            df = self.data_module.get_augmented_history(
                uid=self.uid,
                controls_power_mapping=self.cpm,
                weather_ref=self.weather_ref,
            )

            # Keep only signals required for last state
            df = df.iloc[-self.rl_config.history_length :]
            scaled_df = self.data_module.scale_data(self.uid, df)
            q_factors = self.control_module.get_raw_q_factors(
                uid=self.uid,
                scaled_data=scaled_df,
                rl_config=self.rl_config,
            )
            controls = []

            if isinstance(self.asset, ElectricVehicle):
                trajectories = self.data_module.get_trajectory_data(
                    self.uid, self.index
                )

                df = trajectories[0]

                less_than_60_before_event = (
                    df.loc[df.index[0], "dr_event"] == False
                    and (
                        df.loc[df.index[11], "dr_event"] == True or df.loc[df.index[5], "dr_event"]
                        )
                )

                # Discharge during DR event
                if df.loc[df.index[0], "dr_event"] == True:
                    print("Discharging during DR event")
                    control_int = 0

                # Pre-charge before DR if happening
                elif self.asset.internal_energy < 75 and less_than_60_before_event:
                    print("Pre-charging before DR event")
                    control_int = 2

                # If there is a control buffer, use it
                elif hasattr(self, "control_buffer") and len(self.control_buffer) > 0:
                    print("Using control buffer")
                    print(self.control_buffer)
                    control_int = self.control_buffer.pop(0)
                else:
                    # Iterate over trajectories, and find the one maximizing the cum cost column
                    best_idx = 0
                    best_reward = -99999999
                    for idx, trajectory in enumerate(trajectories):
                        print("Looping over trajectory:")
                        print(trajectory)
                        reward = trajectory["reward_COST"].values.sum()
                        if reward > best_reward:
                            print(f"New best reward: {reward}")
                            best_idx = idx
                            best_reward = reward

                    # Remove NaNs and NA values from list
                    self.control_buffer = [
                        x
                        for x in trajectories[best_idx]["control_1"].values.tolist()
                        if pd.notna(x)
                    ]

                    print("New control buffer:")
                    print(self.control_buffer)

                    control_int = self.control_buffer.pop(0)

                # Should never discharge except in DR
                if df.loc[df.index[0], "dr_event"] == False and control_int == 0:
                    control_int = 1

                # If below 40%, automatically stop discharge
                if self.asset.internal_energy < 30 and control_int == 0:
                    print("Preventing discharge for capacity safety")
                    control_int += 1
                    
                # If under 80% and not in DR, charge
                if self.asset.internal_energy < 75*0.8 and df.loc[df.index[0], "dr_event"] == False:
                    print("Charging to 80% for safety")
                    control_int = 2

                # Return final control
                control = [DiscreteControl(control_int)]
                print(f"Final control: {control}")
                return control

            # HVAC OPTIMIZATION
            if isinstance(self.asset, Building):
                previous_controls = (
                    df.iloc[-2:, :][["control_1", "control_2", "control_3"]]
                    .values[0]
                    .tolist()
                )

                T1 = df.iloc[-1]["temperature_1"]
                T2 = df.iloc[-1]["temperature_2"]
                T3 = df.iloc[-1]["temperature_3"]
                cool_sp = df.iloc[-1]["cool_setpoint"]
                heat_sp = df.iloc[-1]["heat_setpoint"]

                if self.index.minute % 15 != 0:
                    control_int1 = previous_controls[0]
                    control_int2 = previous_controls[1]
                    control_int3 = previous_controls[2]
                    controls = [DiscreteControl(int(c)) for c in previous_controls]
                else:
                    sanitized_q_factors1 = q_factors[0][0, 0:2, :].T.reshape(5, 2)
                    sanitized_q_factors2 = q_factors[1][0, 0:2, :].T.reshape(5, 2)
                    sanitized_q_factors3 = q_factors[2][0, 0:2, :].T.reshape(5, 2)
                    # Switch columns 1 and 2 for each q factor
                    # sanitized_q_factors1[:, [0, 1]] = sanitized_q_factors1[:, [1, 0]]
                    # sanitized_q_factors2[:, [0, 1]] = sanitized_q_factors2[:, [1, 0]]
                    # sanitized_q_factors3[:, [0, 1]] = sanitized_q_factors3[:, [1, 0]]
                    pareto1 = ParetoFrontier(sanitized_q_factors1, op="max")
                    pareto2 = ParetoFrontier(sanitized_q_factors2, op="max")
                    pareto3 = ParetoFrontier(sanitized_q_factors3, op="max")
                    pareto1.initial_position()
                    pareto2.initial_position()
                    pareto3.initial_position()

                    control_int1 = pareto1.get_current_index()
                    control_int2 = pareto2.get_current_index()
                    control_int3 = pareto3.get_current_index()

                    abs_sum_control_int = (
                        abs(control_int1 - 2)
                        + abs(control_int2 - 2)
                        + abs(control_int3 - 2)
                    )

                    while abs_sum_control_int > 3:
                        deltas = [
                            pareto1.delta_to_next_closest(),
                            pareto2.delta_to_next_closest(),
                            pareto3.delta_to_next_closest(),
                        ]

                        if (
                            deltas[0] is not None
                            and (deltas[0] <= deltas[1])
                            and (deltas[0] <= deltas[2])
                        ):
                            pareto1.move_to_next_closest()
                            control_int1 = pareto1.get_current_index()

                        elif (
                            deltas[1] is not None
                            and (deltas[1] <= deltas[0])
                            and (deltas[1] <= deltas[2])
                        ):
                            pareto2.move_to_next_closest()
                            control_int2 = pareto2.get_current_index()

                        elif (
                            deltas[2] is not None
                            and (deltas[2] <= deltas[0])
                            and (deltas[2] <= deltas[1])
                        ):
                            pareto3.move_to_next_closest()
                            control_int3 = pareto3.get_current_index()

                        elif control_int1 != 2:
                            control_int1 = 2

                        elif control_int2 != 2:
                            control_int2 = 2

                        elif control_int3 != 2:
                            control_int3 = 2

                        else:
                            raise ValueError("Incorrect configuation")

                        abs_sum_control_int = (
                            abs(control_int1 - 2)
                            + abs(control_int2 - 2)
                            + abs(control_int3 - 2)
                        )

                    control_int1 = max(control_int1, 2)
                    control_int2 = max(control_int2, 2)
                    control_int3 = max(control_int3, 2)

                    # Zone 1
                    if T1 <= heat_sp + 0.1:
                        control_int1 = 3
                        control_int2 = min(3, control_int2)
                        control_int3 = min(3, control_int3)
                    if T1 >= cool_sp - 0.1:
                        control_int1 = 2

                    # Zone 2
                    if T2 <= heat_sp + 0.1:
                        control_int2 = 3
                        control_int1 = min(3, control_int1)
                        control_int3 = min(3, control_int3)
                    if T2 >= cool_sp - 0.1:
                        control_int2 = 2

                    # Zone 3
                    if T3 <= heat_sp + 0.1:
                        control_int3 = 3
                        control_int1 = min(3, control_int1)
                        control_int2 = min(3, control_int2)
                    if T3 >= cool_sp - 0.1:
                        control_int3 = 2

                    controls.append(DiscreteControl(int(control_int1)))
                    controls.append(DiscreteControl(int(control_int2)))
                    controls.append(DiscreteControl(int(control_int3)))

            else:
                # DEFAULT BEHAVIOR
                for q_tensor in q_factors:
                    # Mean of all objectives
                    q_values = np.mean(q_tensor[-1, :, :], axis=0)

                    # Boltzmann policy selection
                    probabilities = softmax(q_values)

                    # Initialize a mask with all True values
                    mask = np.ones_like(probabilities, dtype=bool)

                    # Do not allow discharge for empty energy storage
                    storage_empty = False
                    if isinstance(self.asset, EnergyStorage):
                        if self.asset.internal_energy <= 0.0:
                            # Set the mask for the first action to False
                            mask[0] = False
                            storage_empty = True

                    # Apply the mask to the probabilities
                    probabilities *= mask

                    # Check if probabilities are all zeros
                    if np.all(probabilities == 0):
                        action_choices = (
                            range(1, len(q_values))
                            if storage_empty
                            else range(len(q_values))
                        )
                        control_int = np.random.choice(action_choices)
                    else:
                        # Normalize probabilities to sum to 1
                        probabilities_sum = np.sum(probabilities)
                        if probabilities_sum == 0:
                            print("Sum of probabilities is zero, cannot normalize")
                            action_choices = (
                                range(1, len(q_values))
                                if storage_empty
                                else range(len(q_values))
                            )
                            control_int = np.random.choice(action_choices)
                        else:
                            probabilities /= probabilities_sum
                            control_int = np.random.choice(
                                range(len(q_values)), p=probabilities
                            )

                    controls.append(DiscreteControl(int(control_int)))

            # Building case
            if "temperature_1" in df.columns:
                temperatures = df.iloc[-1:, :][
                    ["temperature_1", "temperature_2", "temperature_3"]
                ].values.tolist()
                formatted_temperatures = [
                    [f"{value:.2f}" for value in sublist] for sublist in temperatures
                ]
                cool_sp = df.iloc[-1]["cool_setpoint"]
                heat_sp = df.iloc[-1]["heat_setpoint"]
                print(f"  temperature {formatted_temperatures} {(heat_sp,cool_sp)}")
                print(f"  hvac stage: {[c.value - 2 for c in controls]}")

            # Print Rewards
            reward_cols = [c for c in df.columns if "reward" in c]
            return controls
        return None

    # TODO: Add int index compatibility, and parameterize sim time step
    def sample_trajectory(
        self,
        history_df: pd.DataFrame,
        trajectory_len: int,
        time_now: dt.datetime,
        policy: Callable[[pd.DataFrame], int],
    ) -> pd.DataFrame:
        # Work with a copy of the input dataframe
        df = history_df.copy()
        control_cols = [
            CONTROL_KEY + "_" + str(i + 1) for i in range(self.asset.config.n_controls)
        ]

        # Create trajectory sample
        current_idx = time_now
        for _ in range(trajectory_len):
            next_idx = current_idx + dt.timedelta(minutes=5)

            # Augment before to make sure policy has all it needs
            df = self.asset.augment_df(df)
            df = self.data_module.augment_df_with_all(
                uid=self.uid,
                df=df,
                controls_power_mapping=self.cpm,
                weather_ref=self.weather_ref,
            )
            control_values = policy(df)
            df.loc[current_idx, control_cols] = control_values
            df[control_cols] = df[control_cols].astype("Int64")
            df = self.asset.augment_df(df)
            df = self.data_module.augment_df_with_all(
                uid=self.uid,
                df=df,
                controls_power_mapping=self.cpm,
                weather_ref=self.weather_ref,
            )
            df = self.prediction_module.get_model_prediction(
                self.uid, df, current_idx, next_idx
            )

            current_idx = next_idx

        # Update one last time df to avoid NaNs in next state
        df = self.asset.augment_df(df)
        df = self.data_module.augment_df_with_all(
            uid=self.uid,
            df=df,
            controls_power_mapping=self.cpm,
            weather_ref=self.weather_ref,
        )

        # Remove any rows with NaN
        # print("df before NaN drop at end of trajectory")
        # print(df)
        # df = df.dropna()

        return df

    def augment_df_with_q_factors(self):
        pass

    def initialize_q_estimator(self) -> None:
        pass

    def compute_raw_td_error(self) -> None:
        pass

    def compute_td_error(self) -> None:
        pass

    def rl_training(self) -> None:
        # Add real data to RL replay buffer
        # TODO: Add mechanism to ensure there are no date overlaps over time
        history = self.data_module.get_augmented_history(
            self.uid, self.cpm, self.weather_ref
        )

        self._push_data_as_rl_training_data(history)

        # Add simulated data to RL replay buffer
        for idx in history.index:
            trajectories = self.data_module.get_trajectory_data(self.uid, idx)
            for trajectory in trajectories:
                self._push_data_as_rl_training_data(data=trajectory, simulation=True)

        # Add simulated data to RL replay buffer
        product = self.data_module.get_products(self.uid)[0]
        self.control_module.rl_training(self.uid, self.rl_config, self.index, product)

    def _push_data_as_rl_training_data(
        self, data: pd.DataFrame, simulation: bool = False
    ):
        # Scale training data
        data = data.copy()
        scaled_data = self.scale_data(data)

        # Get reward column
        reward_columns = self.data_module.get_product_rewards(self.uid)

        # Get control columns
        control_columns = [col for col in data.columns if col.startswith(CONTROL_KEY)]

        # TODO Add generalized check for this type of circumstance
        # if "hvac_1" in scaled_data.columns:
        #    scaled_data = scaled_data.iloc[:-1]

        # Convert and push data to the replay buffers
        self.control_module.push_data_to_replay_buffers(
            uid=self.uid,
            data=scaled_data,
            seq_len=self.rl_config.history_length,
            rl_config=self.rl_config,
            control_columns=control_columns,
            reward_columns=reward_columns,
            simulation=simulation,
        )

    # -----------------------------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # GENERAL
    # -----------------------------------------------------------------------
    def to_file(self, directory: str = "") -> None:
        if not os.path.isdir(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory, f"agent_{self.uid}.pkl")
        with open(filepath, "wb") as f:
            # It might be necessary to remove non-pickleable attributes or set them to None.
            state = {k: v for k, v in self.__dict__.items() if self._is_pickleable(v)}
            dill.dump(state, f)

    @staticmethod
    def _is_pickleable(value):
        """Helper method to determine if a value is pickleable."""
        try:
            dill.dumps(value)
            return True
        except (dill.PicklingError, AttributeError):
            return False

    @classmethod
    def from_file(cls, uid: UidType, directory: str = "") -> "Agent":
        filepath = os.path.join(directory, f"agent_{uid}.pkl")
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"No saved agent with UID {uid} found at {filepath}"
            )
        with open(filepath, "rb") as f:
            internal_variables = dill.load(f)
        agent = cls(uid)
        agent.__dict__.update(internal_variables)
        return agent

    @classmethod
    def load_or_initialize(
        cls,
        uid: UidType,
        data_module: DataModule,
        control_module: ControlModule,
        prediction_module: PredictionModule,
        config_module: ConfigModule,
        index: IndexType,
        weather_ref: WeatherRef,
        directory: str = "",
    ) -> "Agent":
        filepath = os.path.join(directory, f"agent_{uid}.pkl")
        if os.path.exists(filepath):
            return cls.from_file(uid, directory)
        else:
            return cls(
                uid=uid,
                data_module=data_module,
                control_module=control_module,
                prediction_module=prediction_module,
                config_module=config_module,
                index=index,
                weather_ref=weather_ref,
            )
