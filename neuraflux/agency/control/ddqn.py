import os
from copy import copy
from dataclasses import dataclass

import dill
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import gc


@dataclass
class DDQNPREstimator:
    state_size: int
    action_size: int
    sequence_len: int
    n_rewards: int = 1
    n_controllers: int = 1
    learning_rate: float = 5e-4
    batch_size: int = 500
    discount_factor: float = 0.95

    def __post_init__(self) -> None:
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def __call__(self, *args, **kwargs):
        return self.forward_pass(*args, **kwargs)

    def forward_pass(
        self, states: np.ndarray, target_model: bool = False
    ) -> np.ndarray:
        # Copy state to avoid modifying the original
        states = states.copy()

        # Define model and apply forward pass
        model = self.target_model if target_model else self.model
        outputs = model.predict(states, verbose=0)

        # Ensure outputs are a list, even for a single controller
        if not isinstance(outputs, list):
            outputs = [outputs]

        return outputs

    def compute_td_errors(
        self,
        experience: tuple,
    ) -> list:
        (states, actions, rewards, next_states, dones, errors) = experience

        # Get the current Q-values (list of [batch_size, n_rewards, action_size] for each controller)
        q_values_current = self.forward_pass(states)

        # Get the Q-values for the next states from the target model
        q_values_next_target = self.forward_pass(
            next_states, target_model=True
        )

        # Initialize a list to hold the TD errors for each controller
        td_errors = [
            np.zeros(
                (states.shape[0], self.n_rewards, self.action_size)
            )  # n_samples x n_rewards x n_actions
            for _ in range(self.n_controllers)
        ]

        # Compute TD errors for each controller and reward
        for controller in range(self.n_controllers):
            for reward in range(self.n_rewards):
                # Compute the max Q-value for the next states for each reward
                q_max_next_target = np.max(
                    q_values_next_target[controller][:, reward, :], axis=1
                )

                # Compute the target Q-values
                q_target = rewards[
                    :, reward
                ] + self.discount_factor * q_max_next_target * (1 - dones)

                # Select the corresponding actions for the current controller
                actions_current_controller = actions[:, controller]

                # Compute the TD errors for all actions
                for action in range(self.action_size):
                    is_action_taken = actions_current_controller == action
                    for sample in range(states.shape[0]):
                        if is_action_taken[sample]:
                            td_errors[controller][sample, reward, action] = (
                                q_target[sample]
                                - q_values_current[controller][
                                    sample, reward, action
                                ]
                            )
                        else:
                            td_errors[controller][
                                sample, reward, action
                            ] = 0  # np.nan  # Default value for non-taken actions

        return td_errors

    def compute_td_errors_rmse(self, experience, aggregate=True):
        td_errors = self.compute_td_errors(experience)
        # Check for invalid data and raise ValueError if found
        for controller_td_error in td_errors:
            if controller_td_error.size == 0:
                raise ValueError("Encountered an empty array in TD errors.")

        if aggregate:
            # Compute aggregated RMSE for each controller separately
            rmse_per_controller = [
                np.sqrt(np.nanmean(np.square(controller_td_error)))
                for controller_td_error in td_errors
            ]

            # Aggregate RMSE across all controllers
            aggregated_rmse = np.nanmean(rmse_per_controller)
            return aggregated_rmse
        else:
            # Compute RMSE for each sample individually for each controller
            rmse_per_sample_per_controller = [
                np.sqrt(
                    np.nanmean(np.square(controller_td_error), axis=(1, 2))
                )  # Compute RMSE across rewards and actions
                for controller_td_error in td_errors
            ]

            # Sum the RMSEs element-wise across all controllers and divide by the number of controllers
            averaged_rmse_per_sample = np.nanmean(
                rmse_per_sample_per_controller, axis=0
            )

            # Return the averaged RMSEs for each sample
            return averaged_rmse_per_sample

    def train(
        self,
        experience: tuple,
        replay_buffer_len: int,
        priorities: np.ndarray,
        learning_rate: float = 1e-3,
        beta: float = 0.4,
        n_fit_epochs: int = 5,
    ):
        # Unpack the experience tuple
        (states, actions, rewards, next_states, dones, errors) = experience
        batch_size = states.shape[0]

        # Dimensions of each entry in the experience tuple
        # states is (batch_size, sequence_len, state_size)
        # actions was (batch_size,), is now (batch_size, n_controllers)
        # rewards was (batch_size,), is now (batch_size, n_rewards)
        # next_states is (batch_size, sequence_len, state_size)

        # Copy the states to avoid modifying the original
        states = states.copy()

        # Calculate the necessary targets
        # was (batch_size, n_actions), is now [(batch_size, n_rewards, n_actions), ...]
        # where the len of the list is n_controllers)
        targets = self.forward_pass(states)
        targets_next = self.forward_pass(next_states)
        targets_val = self.forward_pass(next_states, target_model=True)

        for target, target_next, target_val in zip(
            targets, targets_next, targets_val
        ):
            for r in range(self.n_rewards):
                for i in range(batch_size):
                    max_a = np.argmax(target_next[i][r])

                    term_1 = rewards[i][r]
                    term_2 = (
                        self.discount_factor
                        * target_val[i][r][max_a]
                        * (1 - dones[i])
                    )
                    target[i][r][actions[i]] = term_1 + term_2

        # Compute importance sampling weights
        importance_sampling_weights = np.power(
            replay_buffer_len * priorities, -beta
        )
        importance_sampling_weights /= importance_sampling_weights.max()

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=learning_rate, clipnorm=1
            ),
            loss="mse",
        )

        print(f"    Fitting DQN for {n_fit_epochs} epochs with learning rate {learning_rate}")

        self.model.fit(
            tf.convert_to_tensor(states),
            tf.convert_to_tensor(target),
            batch_size=32,
            verbose=0,
            sample_weight=importance_sampling_weights,
            epochs=n_fit_epochs,
        )
        tf.keras.backend.clear_session()
        _ = gc.collect()

    # def train(
    #     self,
    #     experience: tuple,
    #     replay_buffer_len: int,
    #     priorities: np.ndarray,
    #     learning_rate: float = 1e-3,
    #     beta: float = 0.4,
    #     epochs: int = 10,
    # ):
    #     (states, actions, rewards, next_states, dones, errors) = experience

    #     # Copy the states to avoid modifying the original
    #     states = states.copy()

    #     target = self.forward_pass(states)
    #     target_next = self.forward_pass(next_states)
    #     target_val = self.forward_pass(next_states, target_model=True)

    #     for i in range(target.shape[0]):
    #         max_a = np.argmax(target_next[i])

    #         target[i][actions[i]] = rewards[
    #             i
    #         ] + self.discount_factor * target_val[i][max_a] * (1 - dones[i])

    #     # Compute importance sampling weights
    #     importance_sampling_weights = np.power(
    #         replay_buffer_len * priorities, -beta
    #     )
    #     importance_sampling_weights /= importance_sampling_weights.max()

    #     self.model.compile(
    #         optimizer=tf.keras.optimizers.Adam(
    #             learning_rate=learning_rate, clipnorm=1
    #         ),
    #         loss="mse",
    #     )

    #     self.model.fit(
    #         tf.convert_to_tensor(states),
    #         tf.convert_to_tensor(target),
    #         batch_size=32,
    #         verbose=0,
    #         sample_weight=importance_sampling_weights,
    #         epochs=epochs,
    #     )
    #     tf.keras.backend.clear_session()
    #     _ = gc.collect()

    def update_target_model(self) -> None:
        self.target_model.set_weights(self.model.get_weights())

    def to_file(self, directory: str) -> None:
        # Create directory if it doesn't exist
        if not os.path.isdir(directory):
            os.makedirs(directory)

        # Save the model
        model_file = os.path.join(directory, "model.keras")
        self.model.save(model_file)

        # Remove unserializable objects from the attributes
        model_temp = self.model
        target_model_temp = self.target_model
        del self.model
        del self.target_model

        # Save the attributes
        filepath = os.path.join(directory, "attributes.pkl")
        with open(filepath, "wb") as f:
            dill.dump(vars(self), f)

        # Restore attributes
        self.model = model_temp
        self.target_model = target_model_temp

    @classmethod
    def from_file(cls, directory: str = "") -> "DDQNPREstimator":
        """Loads an estimator from the specified directory.

        directory (str): The directory to load the attributes from.
        """
        model_file = os.path.join(directory, "model.keras")
        filepath = os.path.join(directory, "attributes.pkl")
        with open(filepath, "rb") as f:
            internal_variables = dill.load(f)
        self = cls(
            internal_variables["state_size"],
            internal_variables["action_size"],
            internal_variables["sequence_len"],
        )
        self.__dict__.update(internal_variables)
        self.model = tf.keras.models.load_model(model_file)
        self.update_target_model()
        return self

    def copy(self):
        self_copy = copy(self)
        self_copy.model = tf.keras.models.clone_model(self.model)
        self_copy.model.set_weights(self.model.get_weights())
        self_copy.target_model = tf.keras.models.clone_model(self.target_model)
        return self_copy

    # def _build_model(self) -> Model:
    #     # initializer = tf.keras.initializers.Zeros()
    #     input_layer = tf.keras.layers.Input(
    #         shape=(self.sequence_len, self.state_size)
    #     )

    #     x = tf.keras.layers.LSTM(
    #         128, activation="tanh", return_sequences=False
    #     )(input_layer)

    #     q_values_list = [
    #         tf.keras.layers.Dense(
    #             self.action_size,
    #             activation="linear",
    #         )(x)
    #         for _ in range(self.n_rewards)
    #     ]

    #     model = Model(inputs=[input_layer], outputs=q_values_list)
    #     model.compile(
    #         optimizer=tf.keras.optimizers.Adam(
    #             learning_rate=self.learning_rate, clipnorm=1
    #         ),
    #         loss="mse",
    #     )

    #     return model

    def _build_model(self) -> Model:
        input_layer = tf.keras.layers.Input(
            shape=(self.sequence_len, self.state_size)
        )

        # Shared layers
        x = tf.keras.layers.GRU(
            128, activation="tanh", return_sequences=False
        )(input_layer)

        # Output layer for all actions and rewards
        output_layers = []
        for _ in range(self.n_controllers):
            output_layer = tf.keras.layers.Dense(
                self.n_rewards * self.action_size, activation="linear"
            )(x)
            # Reshape to [number of rewards, number of actions]
            output_reshaped = tf.keras.layers.Reshape(
                (self.n_rewards, self.action_size)
            )(output_layer)
            output_layers.append(output_reshaped)

        model = Model(inputs=[input_layer], outputs=output_layers)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate, clipnorm=1
            ),
            loss="mse",
        )

        return model
