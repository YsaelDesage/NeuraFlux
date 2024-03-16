import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import patch
from neuraflux.agency.control.ddqn import DDQNPREstimator


def test_post_init():
    state_size, action_size, sequence_len = 10, 5, 3
    estimator = DDQNPREstimator(state_size, action_size, sequence_len)

    assert estimator.model is not None, "Model should be initialized"
    assert (
        estimator.target_model is not None
    ), "Target model should be initialized"

    model_weights = estimator.model.get_weights()
    target_model_weights = estimator.target_model.get_weights()
    for mw, tmw in zip(model_weights, target_model_weights):
        assert np.array_equal(
            mw, tmw
        ), "Weights should be the same after post init"


def test_forward_pass():
    # Initialize the estimator with dummy parameters
    state_size, action_size, sequence_len, n_rewards, n_agents = 10, 5, 3, 2, 2
    estimator = DDQNPREstimator(
        state_size, action_size, sequence_len, n_rewards, n_agents
    )

    # Create dummy states as input
    dummy_states = np.random.random(
        (4, sequence_len, state_size)
    )  # Batch size of 4

    # Perform a forward pass using the model
    outputs = estimator.forward_pass(dummy_states)

    # Check if the output is a list
    assert isinstance(outputs, list), "Output should be a list"

    # Check the length of the output list (should be equal to the number of agents)
    assert (
        len(outputs) == n_agents
    ), f"Output list should have {n_agents} elements"

    # Check the shape of each output in the list
    for output in outputs:
        assert output.shape == (
            4,
            n_rewards,
            action_size,
        ), "Output shape is incorrect for each agent"


def test_training_with_dummy_data():
    state_size, action_size, sequence_len, n_rewards, n_controllers = (
        10,
        5,
        3,
        4,
        2,
    )
    estimator = DDQNPREstimator(
        state_size, action_size, sequence_len, n_rewards, n_controllers
    )

    # Create dummy data
    batch_size = 8
    states = np.random.random((batch_size, sequence_len, state_size))
    actions = np.random.randint(0, action_size, (batch_size, n_controllers))
    rewards = np.random.random((batch_size, n_rewards))
    next_states = np.random.random((batch_size, sequence_len, state_size))
    dones = np.random.randint(0, 2, (batch_size,)).astype(bool)
    errors = np.random.random((batch_size,))

    # Create a mock replay buffer
    replay_buffer_len = 1000
    priorities = np.random.random((batch_size,))

    # Train the model
    estimator.train(
        (states, actions, rewards, next_states, dones, errors),
        replay_buffer_len,
        priorities,
    )

    # Optionally, assert changes in weights, reduced loss, etc.


def test_compute_td_errors():
    state_size, action_size, sequence_len, n_rewards, n_controllers = (
        10,
        5,
        3,
        4,
        2,
    )
    estimator = DDQNPREstimator(
        state_size, action_size, sequence_len, n_rewards, n_controllers
    )

    # Create dummy data
    batch_size = 8
    states = np.random.random((batch_size, sequence_len, state_size))
    actions = np.random.randint(0, action_size, (batch_size, n_controllers))
    rewards = np.random.random((batch_size, n_rewards))
    next_states = np.random.random((batch_size, sequence_len, state_size))
    dones = np.random.randint(0, 2, (batch_size,)).astype(bool)

    # Compute TD errors
    td_errors = estimator.compute_td_errors(
        (states, actions, rewards, next_states, dones, None)
    )

    # Assertions
    assert td_errors is not None, "TD errors should not be None"
    assert isinstance(td_errors, list), "TD errors should be a list"
    assert (
        len(td_errors) == n_controllers
    ), "TD errors list should have one array per controller"

    for controller_td_error in td_errors:
        assert isinstance(
            controller_td_error, np.ndarray
        ), "Each controller's TD error should be a numpy array"
        assert controller_td_error.shape == (
            batch_size,
            n_rewards,
            action_size,
        ), "TD error array shape should be (batch_size, n_rewards, n_actions)"


def test_compute_td_errors_rmse_aggregate():
    state_size, action_size, sequence_len, n_rewards, n_controllers = (
        10,
        5,
        3,
        4,
        2,
    )
    estimator = DDQNPREstimator(
        state_size, action_size, sequence_len, n_rewards, n_controllers
    )

    # Create dummy data
    batch_size = 8
    states = np.random.random((batch_size, sequence_len, state_size))
    actions = np.random.randint(0, action_size, (batch_size, n_controllers))
    rewards = np.random.random((batch_size, n_rewards))
    next_states = np.random.random((batch_size, sequence_len, state_size))
    dones = np.random.randint(0, 2, (batch_size,)).astype(bool)

    # Compute TD errors RMSE with aggregate = True
    td_errors_rmse_aggregate = estimator.compute_td_errors_rmse(
        (states, actions, rewards, next_states, dones, None), aggregate=True
    )

    # Assertions for aggregated case
    assert (
        td_errors_rmse_aggregate is not None
    ), "Aggregated TD errors RMSE should not be None"
    assert isinstance(
        td_errors_rmse_aggregate, float
    ), "Aggregated TD errors RMSE should be a float"


def test_compute_td_errors_rmse_individual():
    state_size, action_size, sequence_len, n_rewards, n_controllers = (
        10,
        5,
        3,
        4,
        2,
    )
    estimator = DDQNPREstimator(
        state_size, action_size, sequence_len, n_rewards, n_controllers
    )

    # Create dummy data
    batch_size = 8
    states = np.random.random((batch_size, sequence_len, state_size))
    actions = np.random.randint(0, action_size, (batch_size, n_controllers))
    rewards = np.random.random((batch_size, n_rewards))
    next_states = np.random.random((batch_size, sequence_len, state_size))
    dones = np.random.randint(0, 2, (batch_size,)).astype(bool)

    # Compute TD errors RMSE with aggregate = False
    td_errors_rmse_individual = estimator.compute_td_errors_rmse(
        (states, actions, rewards, next_states, dones, None), aggregate=False
    )

    # Assertions for individual case
    assert (
        td_errors_rmse_individual is not None
    ), "Individual TD errors RMSE should not be None"
    assert isinstance(
        td_errors_rmse_individual, np.ndarray
    ), "Individual TD errors RMSE should be a numpy array"
    assert td_errors_rmse_individual.shape == (
        batch_size,
    ), "Individual TD errors RMSE array shape should be (batch_size,)"
