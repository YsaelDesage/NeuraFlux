import datetime as dt
import json
import os
import random
from copy import deepcopy
import traceback
import dill
import numpy as np
import pandas as pd
import tensorflow as tf

from neuraflux.agency.agent import Agent
from neuraflux.agency.config.config_module import ConfigModule
from neuraflux.agency.control.control_module import ControlModule
from neuraflux.agency.data.data_module import DataModule
from neuraflux.agency.prediction.prediction_module import (
    PredictionModel,
    PredictionModule,
)
from neuraflux.assets.building import (
    Building,
    get_simulation_properties_for_building,
)
from neuraflux.assets.energy_storage import EnergyStorage
from neuraflux.assets.electric_vehicle import ElectricVehicle
from neuraflux.global_variables import DT_STR_FORMAT
from neuraflux.schemas.config import (
    BuildingConfig,
    ElectricVehicleConfig,
    EnergyStorageConfig,
    ModelEvaluationMetrics,
    SimulationGlobalConfig,
)
from neuraflux.time_ref import TimeRef
from neuraflux.weather_ref import WeatherRef


class ElectricVehicleModel(PredictionModel):
    def train(self, historical_signals: pd.DataFrame) -> None:
        pass

    def predict(
        self,
        signals_input: pd.DataFrame,
        current_index: int | dt.datetime,
        new_index: int | dt.datetime,
    ) -> pd.DataFrame:
        E = signals_input.loc[signals_input.index[-1], "internal_energy"]
        u: int = signals_input.loc[signals_input.index[-1], "control_1"]
        availability = signals_input.loc[signals_input.index[-1], "availability"]
        power_map = {0: -25, 1: 0, 2: 25}
        power = power_map[u]
        dE = power * 300 / 3600

        if availability == 0:
            dE = -10

        output_df = signals_input.copy()
        new_E = E + dE
        new_E = min(75, new_E)
        new_E = max(0, new_E)
        output_df.loc[new_index, "internal_energy"] = new_E

        return output_df

    def evaluate(self, historical_signals: pd.DataFrame) -> ModelEvaluationMetrics:
        return ModelEvaluationMetrics(
            root_mean_squared_error=0,
            mean_absolute_error=0,
            mean_squared_error=0,
        )


class EnergyStorageModel(PredictionModel):
    def train(self, historical_signals: pd.DataFrame) -> None:
        pass

    def predict(
        self,
        signals_input: pd.DataFrame,
        current_index: int | dt.datetime,
        new_index: int | dt.datetime,
    ) -> pd.DataFrame:
        E = signals_input.loc[signals_input.index[-1], "internal_energy"]
        u: int = signals_input.loc[signals_input.index[-1], "control_1"]
        power_map = {0: -100, 1: 0, 2: 100}
        power = power_map[u]
        dE = power * 300 / 3600

        output_df = signals_input.copy()
        new_E = E + dE
        new_E = min(500, new_E)
        new_E = max(0, new_E)
        output_df.loc[new_index, "internal_energy"] = new_E

        return output_df

    def evaluate(self, historical_signals: pd.DataFrame) -> ModelEvaluationMetrics:
        return ModelEvaluationMetrics(
            root_mean_squared_error=0,
            mean_absolute_error=0,
            mean_squared_error=0,
        )


class BuildingModel(PredictionModel):
    def train(self, historical_signals: pd.DataFrame) -> None:
        pass

    def predict(
        self,
        signals_input: pd.DataFrame,
        current_index: int | dt.datetime,
        new_index: int | dt.datetime,
    ) -> pd.DataFrame:
        df = signals_input.copy()

        oat = df.loc[df.index[-1], "outside_air_temperature"]
        temp1 = df.loc[current_index, "temperature_1"]
        temp2 = df.loc[current_index, "temperature_2"]
        temp3 = df.loc[current_index, "temperature_3"]
        temp = np.array([temp1, temp2, temp3])
        u1 = df.loc[df.index[-1], "control_1"]
        u2 = df.loc[df.index[-1], "control_2"]
        u3 = df.loc[df.index[-1], "control_3"]
        control = [u1, u2, u3]
        power_map = {
            0: 20,
            1: 10,
            2: 0,
            3: 10,
            4: 20,
        }
        power_vec = [abs(power_map[int(c)]) for c in control]

        Uinv, F, C = get_simulation_properties_for_building(5)

        # HVAC Q input (in Watts)
        Q_hvac = np.array(power_vec) * 1000

        # Outside heat input (e.g. Solar)
        Q_in = np.zeros(len(Uinv))  # None for now

        # Get total Q contribution (include HVAC by default)
        term1 = np.dot(F, oat)
        term2 = np.multiply(C.T / (5 * 60), temp).flatten()
        Q = Q_hvac + Q_in + term1 + term2

        temperatures = np.dot(Q, Uinv).diagonal().tolist()

        output_df = signals_input.copy()
        T1 = temperatures[0]
        T2 = temperatures[1]
        T3 = temperatures[2]
        output_df.loc[new_index, "temperature_1"] = T1
        output_df.loc[new_index, "temperature_2"] = T2
        output_df.loc[new_index, "temperature_3"] = T3

        return output_df

    def evaluate(self, historical_signals: pd.DataFrame) -> ModelEvaluationMetrics:
        return ModelEvaluationMetrics(
            root_mean_squared_error=0,
            mean_absolute_error=0,
            mean_squared_error=0,
        )


class Simulation:
    def __init__(self, simulation_config: SimulationGlobalConfig) -> None:
        # Fix seeds
        SEED_VALUE = 42
        np.random.seed(SEED_VALUE)
        random.seed(SEED_VALUE)
        tf.random.set_seed(SEED_VALUE)

        # Store configuration
        self.config = simulation_config
        self.directory = simulation_config.directory

        # Global time variables
        self.start_time = dt.datetime.strptime(
            self.config.time.start_time,
            DT_STR_FORMAT,
        )
        self.end_time = dt.datetime.strptime(
            self.config.time.end_time,
            DT_STR_FORMAT,
        )

        # Initialization process
        try:
            self._initialize_modules_and_references()
            self._initialize_assets()
            self._initialize_agents()
        except:
            print(traceback.format_exc())
            print("Unable to initialize simulation, must load from directory.")

    def run(self) -> None:
        # Simulation time loop
        n_rl_trainings = 0
        while self.time_ref.get_time_utc() <= self.end_time:
            # Outside air temperature
            outside_air_temperature = self.weather_ref.get_temperature_at_time(
                self.index
            )

            for agent in self.agents:
                agent.update_time_reference(self.index)
                agent.update_modules(
                    data_module=self.modules["data"],
                    control_module=self.modules["control"],
                    prediction_module=self.modules["prediction"],
                    config_module=self.modules["config"],
                    weather_ref=self.weather_ref,
                )

            # Calculate elapsed time in minutes between start and index
            elapsed_minutes = (self.index - self.start_time).total_seconds() / 60
            a_day_in_mn = 1440
            print(f"Time: {self.index}")
            # NOTE: The order is important - Agents, assets then time

            agent = self.agents[0]

            # Run agent to collect data and run policy eval
            agent.asset_data_collection()
            agent.policy_evaluation(time_now=self.index)

            # Get control from agent
            agent_ctrl = agent.get_controls()

            # Save simulation every day
            if elapsed_minutes % a_day_in_mn == 0 and elapsed_minutes != 0:
                print("saving simulation")
                self.save()

            # Train every week
            if (
                elapsed_minutes % (a_day_in_mn * 1) == 0
                # elapsed_minutes % 120 == 0
                and elapsed_minutes != 0
                and n_rl_trainings < 3
            ):
                print("RL TRAINING !")
                agent.fit_scalers_on_data()  # Will only fit the first time
                agent.rl_training()
                n_rl_trainings += 1
                print("Saving simulation")
                self.save()

            # Apply control (if any) to asset
            asset = self.assets[0]
            shadow_asset = self.shadow_assets[0]
            if agent_ctrl is None:
                ctrl = asset.get_auto_control(self.index, outside_air_temperature)
                shadow_ctrl = ctrl
            else:
                ctrl = agent_ctrl
                shadow_ctrl = shadow_asset.get_auto_control(
                    self.index, outside_air_temperature
                )

            # Increment simulation time
            self.time_ref.increment_time(self.config.time.step_size_s)

            # Time variables update
            self.index = self.time_ref.get_time_utc()
            asset.step(ctrl, self.index, outside_air_temperature)
            shadow_asset.step(shadow_ctrl, self.index, outside_air_temperature)

    def save(self) -> None:
        save_directory = os.path.join(self.config.directory)
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        # Save config as JSON
        try:
            config_file = os.path.join(save_directory, "config.json")
            with open(config_file, "w") as f:
                json.dump(
                    self.config.model_dump(),
                    f,
                    ensure_ascii=False,
                    indent=4,
                )
        except Exception as e:
            print(f"Error saving config: \n {e}")

        # Save modules
        for module in ["data", "config", "control", "prediction"]:
            try:
                self.modules[module].to_file(save_directory)
            except Exception as e:
                print(f"Error saving {module} module: \n {e}")

        # Save references
        self.time_ref.to_file(save_directory)
        self.weather_ref.to_file(save_directory)

        # Save agents
        agent_save_dir = os.path.join(save_directory, "agents")
        for agent in self.agents:
            agent.to_file(agent_save_dir)

        # Save shadow assets
        shadow_asset_save_dir = os.path.join(save_directory, "shadow_assets.pkl")
        with open(shadow_asset_save_dir, "wb") as f:
            dill.dump(self.shadow_assets, f)

    def load(self, save_directory=None) -> None:
        all_saves = os.listdir(self.config.directory)
        if save_directory is None:
            save_directory = os.path.join(self.config.directory, all_saves[-1])
        if os.path.isdir(save_directory):
            # Load modules
            self.modules = {}
            self.modules["data"] = DataModule.from_file(save_directory)
            self.modules["config"] = ConfigModule.from_file(save_directory)
            self.modules["control"] = ControlModule.from_file(save_directory)
            self.modules["prediction"] = PredictionModule.from_file(save_directory)

            # Load references
            # self.time_ref = TimeRef.from_file(save_directory)
            # self.weather_ref = WeatherRef.from_file(save_directory)

            # Load agents
            # agent_save_dir = os.path.join(save_directory, "agents")
            # self.agents = [Agent.from_file(uid="1", directory=agent_save_dir)]

            # Load shadow assets
            shadow_asset_save_dir = os.path.join(save_directory, "shadow_assets.pkl")
            with open(shadow_asset_save_dir, "rb") as f:
                self.shadow_assets = dill.load(f)

    @classmethod
    def from_directory(cls, directory: str) -> None:
        config_file = os.path.join(directory, "config.json")
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        return cls(SimulationGlobalConfig(**config_dict))

    def _initialize_modules_and_references(self) -> None:
        # References
        self.time_ref = TimeRef.load_or_initialize(start_time_utc=self.start_time)
        self.index = self.time_ref.get_time_utc()
        self.weather_ref = WeatherRef.load_or_initialize(
            lat=self.config.geography.location_lat,
            lon=self.config.geography.location_lon,
            alt=self.config.geography.location_alt,
            start=self.start_time,
            end=self.end_time,
        )

        # Modules
        self.data_module = DataModule.load_or_initialize()
        self.config_module = ConfigModule.load_or_initialize()
        self.prediction_module = PredictionModule.load_or_initialize(self.directory)
        self.control_module = ControlModule.load_or_initialize()

        self.modules = {
            "config": self.config_module,
            "control": self.control_module,
            "data": self.data_module,
            "prediction": self.prediction_module,
        }

    def _initialize_assets(self) -> None:
        self.assets = []
        timestamp = self.time_ref.get_time_utc()
        oat = self.weather_ref.get_temperature_at_time(timestamp)
        for asset_uid, asset_config in self.config.assets.items():
            # Add outside air temperature to asset config
            asset_config.initial_state_dict[
                "outside_air_temperature"
            ] = self.weather_ref.get_temperature_at_time(self.index)

            if isinstance(asset_config, EnergyStorageConfig):
                asset = EnergyStorage.load_or_initialize(
                    asset_uid, self.directory, asset_config, timestamp, oat
                )
            elif isinstance(asset_config, BuildingConfig):
                asset = Building.load_or_initialize(
                    asset_uid, self.directory, asset_config, timestamp, oat
                )
            elif isinstance(asset_config, ElectricVehicleConfig):
                asset = ElectricVehicle.load_or_initialize(
                    asset_uid, self.directory, asset_config, timestamp, oat
                )
            else:
                raise ValueError(f"Asset type not supported: {type(asset_config)}.")
            self.assets.append(asset)
        self.shadow_assets = [deepcopy(asset) for asset in self.assets]

    def _initialize_agents(self) -> None:
        self.agents = []
        for asset, shadow_asset in zip(self.assets, self.shadow_assets):
            # Define important variables related to the asset
            uid = asset.name

            # Push agent config to config module
            # TODO: Check if one already exists first
            self.config_module.push_new_agent_config(
                uid=uid, agent_config=self.config.agent_config
            )

            agent = Agent.load_or_initialize(
                uid=uid,
                data_module=self.data_module,
                control_module=self.control_module,
                prediction_module=self.prediction_module,
                config_module=self.config_module,
                index=self.time_ref.get_time_utc(),
                weather_ref=self.weather_ref,
            )
            agent.assign_to_asset(asset)
            agent.assign_to_shadow_asset(shadow_asset)

            # TODO: Initialize this in agent
            signal_inputs = self.config.agent_config.prediction.signal_inputs
            signal_outputs = self.config.agent_config.prediction.signal_outputs

            if isinstance(asset, EnergyStorage):
                model = EnergyStorageModel(signal_inputs, signal_outputs)  # type: ignore
            elif isinstance(asset, Building):
                model = BuildingModel(signal_inputs, signal_outputs)
            elif isinstance(asset, ElectricVehicle):
                model = ElectricVehicleModel(signal_inputs, signal_outputs)
            self.modules["prediction"].assign_model(  # type: ignore
                uid=uid, model=model
            )

            self.agents.append(agent)
