import numpy as np
import pandas as pd
from typing import Any
from neuraflux.agency.control.ddqn import DDQNPREstimator
from neuraflux.agency.control.replay_buffer import ReplayBuffer
from neuraflux.agency.module import Module
from neuraflux.local_typing import UidType, IndexType
from neuraflux.schemas.config import RLConfig
from neuraflux.global_variables import PRODUCTION_KEY
from neuraflux.agency.control.control_utils import (
    convert_data_to_experience,
    convert_data_to_state,
)
from neuraflux.agency.control.rl_training import (
    inner_simulation_training_loop,
    final_tuning_loop,
)
from copy import copy


class ControlModule(Module):
    def rl_training(
        self, uid: UidType, rl_config: RLConfig, index: str, product
    ) -> None:
        # Initialize buffers and Q estimators if necessary
        self._initialize_agent_replay_buffers_if_necessary(uid)
        self._initialize_q_estimator_if_necessary(uid, rl_config, product)

        # Retrieve necessary components
        real_buffer = self.real_replay_buffers[uid]
        sim_buffer = self.sim_replay_buffers[uid]
        print(f"Len of real | sim buffers: {len(real_buffer)} | {len(sim_buffer)}")
        q_estimator = self.get_model_from_registry(uid, PRODUCTION_KEY)

        real_td_errors_list = []
        sim_td_errors_list = []

        max_iterations = 25
        target_net_update_freq = 4
        target_update_err_thresh = 0.01

        for i in range(max_iterations):
            print(f"• {i}/{max_iterations}")
            # Update
            if i % target_net_update_freq == 0:
                print("  UPDATING TARGET NETWORK WEIGHTS!")
                q_estimator.update_target_model()

            (
                q_estimator,
                real_td_errors,
                sim_td_errors,
            ) = inner_simulation_training_loop(
                real_buffer,
                sim_buffer,
                q_estimator,
                sampling_size=rl_config.experience_sampling_size,
                learning_rate=rl_config.learning_rate,
                n_fit_epochs=rl_config.n_fit_epochs,
            )

            # Store TD errors from last inner training
            # real_td_errors_list.append(real_td_errors)
            # sim_td_errors_list.append(sim_td_errors)

            # Check if early stopping conditions were encountered
            if i % target_net_update_freq == 0:
                if real_td_errors[0] <= target_update_err_thresh:
                    break

            # Update production model
            self.push_new_model_to_registry(uid, PRODUCTION_KEY, q_estimator)

            # Break logic - if the mean of the last 10 TD errors is less than
            # 10% of the mean of the first 50 TD errors, then break the loop
            # if i >= min_iterations:
            #     all_first_errors = [e[0] for e in real_td_errors_list]
            #     mean_head = np.mean(all_first_errors[0:min_iterations])
            #     mean_tail = np.mean(all_first_errors[-10:])
            #     print(f"Mean head: {mean_head} | Mean tail: {mean_tail}")
            #     if mean_tail / mean_head <= 0.1:
            #         break
            print()

        # Final tuning
        (
            q_estimator,
            real_td_errors,
            sim_td_errors,
        ) = final_tuning_loop(
            replay_buffer_real=real_buffer,
            replay_buffer_sim=sim_buffer,
            q_estimator=q_estimator,
            learning_rate=rl_config.learning_rate,
            n_fit_epochs=rl_config.n_fit_epochs,
        )

        # Store TD errors from last inner training
        real_td_errors_list.append(real_td_errors)
        sim_td_errors_list.append(sim_td_errors)

        # Update production model
        self.push_new_model_to_registry(uid, PRODUCTION_KEY, q_estimator)

        # Flag this UID as being ready to use RL
        if uid not in self.rl_model_ready_list:
            self.rl_model_ready_list.append(uid)

        # Store this iteration's training logs
        self.push_new_rl_training_log(
            uid,
            index,
            {
                "real_td_errors": real_td_errors_list,
                "sim_td_errors": sim_td_errors_list,
            },
        )

    def push_new_rl_training_log(
        self, uid: UidType, index: IndexType, logs: Any
    ) -> None:
        self._initialize_rl_training_logs_if_necessary(uid)
        self.rl_training_logs[uid][index] = logs

    def get_rl_training_logs(
        self, uid: UidType, index=None
    ) -> dict[IndexType, Any] | Any:
        if index is None:
            return self.rl_training_logs[uid]
        return self.rl_training_logs[uid][index]

    # -----------------------------------------------------------------------
    # Q-FACTORS
    # -----------------------------------------------------------------------
    def get_raw_q_factors(
        self,
        uid: UidType,
        scaled_data: pd.DataFrame,
        rl_config: RLConfig,
    ) -> None:
        # Work with a copy of the input data
        scaled_df = scaled_data.copy()

        # Convert the dataframe to its NumPy representation
        seq_len = rl_config.history_length
        state_columns = self._get_state_columns_from_rl_config(rl_config)
        states = np.array(convert_data_to_state(scaled_df, state_columns, seq_len))

        # Apply the Q estimator on the states to get the Q factors
        q_estimator = self.get_model_from_registry(uid, PRODUCTION_KEY)
        q_factors = q_estimator.forward_pass(states)

        return q_factors

    def augment_df_with_q_factors(
        self,
        uid: UidType,
        data: pd.DataFrame,
        scaled_data: pd.DataFrame,
        rl_config: RLConfig,
    ) -> None:
        # Work with a copy of the input data
        df = data.copy()
        scaled_df = scaled_data.copy()

        # Convert the dataframe to its NumPy representation
        seq_len = rl_config.history_length
        state_columns = self._get_state_columns_from_rl_config(rl_config)
        states = np.array(convert_data_to_state(scaled_df, state_columns, seq_len))

        # Apply the Q estimator on the states to get the Q factors
        q_estimator = self.get_model_from_registry(uid, PRODUCTION_KEY)
        q_factors = q_estimator.forward_pass(states)

        # Add Q-factors to the dataframe and return
        # NOTE: q_factors are a list (n_controllers len) with elements of
        # dimension (seq_len, n_actions, n_rewards)
        for c, q_vals in enumerate(q_factors):
            for r in range(q_vals.shape[1]):
                q_cols = [f"Q{r+1}_C{c+1}_U{i+1}" for i in range(q_vals.shape[2])]
                df.loc[df.index.values[seq_len - 1 :], q_cols] = q_vals[:, r, :]
        return df

    def is_rl_model_ready(self, uid: UidType) -> bool:
        return uid in self.rl_model_ready_list

    # -----------------------------------------------------------------------
    # REPLAY BUFFERS
    # -----------------------------------------------------------------------
    def push_data_to_replay_buffers(
        self,
        uid: UidType,
        data: pd.DataFrame,
        seq_len: int,
        rl_config: RLConfig,
        control_columns: list[str],
        reward_columns: list[str],
        simulation: bool = False,
    ) -> None:
        # Initialize replay buffer if necessary
        self._initialize_agent_replay_buffers_if_necessary(uid)

        # Use the simulation or normal replay buffer based on the input flag
        if simulation:
            replay_buffer = self.sim_replay_buffers[uid]
        else:
            replay_buffer = self.real_replay_buffers[uid]

        # Convert the dataframe to its NumPy representation
        state_columns = self._get_state_columns_from_rl_config(rl_config)
        experience_batch = convert_data_to_experience(
            data, seq_len, state_columns, control_columns, reward_columns
        )

        # Add the experience samples from the dataframe to the replay buffer
        for experience in zip(*experience_batch):
            replay_buffer.add_experience_sample(experience=experience)

    # -----------------------------------------------------------------------
    # Q ESTIMATORS
    # -----------------------------------------------------------------------
    def initialize_agent_q_estimator_from_config(
        self, uid: UidType, rl_config: RLConfig, product
    ) -> None:
        # Define important variables
        state_columns = self._get_state_columns_from_rl_config(rl_config)
        state_size = len(state_columns)

        # Initialize dictionary if first time using UID
        if uid not in self.rl_model_registry.keys():
            self.rl_model_registry[uid] = {}

        # Initialize Q estimator
        q_estimator = DDQNPREstimator(
            state_size=state_size,
            action_size=rl_config.action_size,
            sequence_len=rl_config.history_length,
            n_rewards=len(product.get_reward_names()),
            n_controllers=rl_config.n_controllers,
            learning_rate=rl_config.learning_rate,
            batch_size=rl_config.batch_size,
            discount_factor=rl_config.discount_factor,
        )
        self.push_new_model_to_registry(uid, PRODUCTION_KEY, q_estimator)

    # -----------------------------------------------------------------------
    # MODEL REGISTRY
    # -----------------------------------------------------------------------
    def push_new_model_to_registry(self, uid: UidType, key: str, model: Any) -> None:
        # Initialize dictionary if first time using UID
        if uid not in self.rl_model_registry.keys():
            self.rl_model_registry[uid] = {}

        # Push model to registry
        self.rl_model_registry[uid][key] = model

    def get_model_from_registry(self, uid: UidType, key: str) -> Any:
        return self.rl_model_registry[uid][key]

    # -----------------------------------------------------------------------
    # INTERNAL
    # -----------------------------------------------------------------------
    def _initialize_data_structures(self) -> None:
        self.rl_model_registry: dict[UidType, dict[str, DDQNPREstimator]] = {}
        self.real_replay_buffers: dict[UidType, ReplayBuffer] = {}
        self.sim_replay_buffers: dict[UidType, ReplayBuffer] = {}
        self.rl_model_ready_list: list[UidType] = []
        self.rl_training_logs: dict[UidType, dict[IndexType, Any]] = {}

    def _initialize_rl_training_logs_if_necessary(self, uid: UidType) -> None:
        if uid not in self.rl_training_logs.keys():
            self.rl_training_logs[uid] = {}

    def _initialize_agent_replay_buffers_if_necessary(self, uid: UidType) -> None:
        # Initialize real replay buffer if necessary
        if uid not in self.real_replay_buffers.keys():
            self.real_replay_buffers[uid] = ReplayBuffer()

        # Initialize simulation replay buffer if necessary
        if uid not in self.sim_replay_buffers.keys():
            self.sim_replay_buffers[uid] = ReplayBuffer()

    def _initialize_q_estimator_if_necessary(
        self, uid: UidType, rl_config: RLConfig, product
    ) -> None:
        # Initialize Q estimator if necessary
        if uid not in self.rl_model_registry.keys():
            self.initialize_agent_q_estimator_from_config(uid, rl_config, product)

    def _get_state_columns_from_rl_config(self, rl_config) -> list[str]:
        # Work with a copy of the state signals list
        all_state_signals = copy(rl_config.state_signals)

        # Add time features to state signals if necessary
        if rl_config.add_hourly_time_features_to_state:
            all_state_signals += ["tf_cos_h", "tf_sin_h"]
        if rl_config.add_daily_time_features_to_state:
            all_state_signals += ["tf_cos_d", "tf_sin_d"]
        if rl_config.add_weekly_time_features_to_state:
            all_state_signals += ["tf_cos_w", "tf_sin_w"]
        if rl_config.add_monthly_time_features_to_state:
            all_state_signals += ["tf_cos_m", "tf_sin_m"]

        return all_state_signals
