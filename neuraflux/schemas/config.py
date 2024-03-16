from typing import Any, Dict, List, Optional, Union
from enum import Enum, unique
from pydantic import field_validator

from neuraflux.global_variables import (
    CONTROL_KEY,
    POWER_KEY,
    TIMESTAMP_KEY,
    OAT_KEY,
)
from neuraflux.local_typing import UidType
from .base import BaseSchema


# ----------------------------------------------------------------------------
# ASSET CONFIG
# ----------------------------------------------------------------------------
class AssetConfig(BaseSchema):
    core_variables: List[str] = [
        TIMESTAMP_KEY,
        CONTROL_KEY,
        POWER_KEY,
        OAT_KEY,
    ]
    initial_state_dict: Dict[str, Any] = {}
    n_controls: int = 1


# BUILDING ASSET
class BuildingConfig(AssetConfig):
    control_power_mapping: Dict[int, float] = {
        0: 20,
        1: 10,
        2: 0,
        3: 10,
        4: 20,
    }
    tracked_variables: List[str] = [
        "temperature",
        "hvac",
        "cool_setpoint",
        "heat_setpoint",
        "occupancy",
    ]
    n_controls: int = 3
    initial_state_dict: Dict[str, Any] = {
        "temperature": [21.0, 21.0, 21.0],
        "hvac": [0, 0, 0],
    }
    dt: int = 5
    occ_times: tuple[int, int] = (8, 18)
    occ_setpoints: tuple[float, float] = (20.0, 22.0)
    unocc_setpoints: tuple[float, float] = (16.0, 26.0)


# ENERGY STORAGE ASSET
class EnergyStorageEfficiencyLimits:
    UPPER_LIMIT: float = 1.0
    LOWER_LIMIT: float = 0.0


class EnergyStorageEfficiency(BaseSchema):
    value: float = 1.0

    @field_validator("value")
    def check_value(cls, v: float) -> float:  # pylint: disable=no-self-argument
        if (
            v > EnergyStorageEfficiencyLimits.UPPER_LIMIT
            or v < EnergyStorageEfficiencyLimits.LOWER_LIMIT
        ):
            raise ValueError("Efficiency must be between 0 and 1.")
        return v


class EnergyStorageConfig(AssetConfig):
    capacity_kwh: float = 500.0
    control_power_mapping: Dict[int, float] = {1: -100, 2: 0, 3: 100}
    efficiency_in: EnergyStorageEfficiency = EnergyStorageEfficiency()
    efficiency_out: EnergyStorageEfficiency = EnergyStorageEfficiency()
    decay_factor: float = 1.0
    tracked_variables: List[str] = ["internal_energy"]


class ElectricVehicleConfig(AssetConfig):
    capacity_kwh: float = 75.0
    control_power_mapping: Dict[int, float] = {0: -25, 1: 0, 2: 25}
    tracked_variables: List[str] = ["internal_energy", "availability"]
    initial_state_dict: Dict[str, Any] = {"internal_energy": 75, "availability": 1}

# ----------------------------------------------------------------------------
# PREDICTION MODULE CONFIGS
# ----------------------------------------------------------------------------
class ModelEvaluationMetrics(BaseSchema):
    root_mean_squared_error: float
    mean_absolute_error: float
    mean_squared_error: float


# ----------------------------------------------------------------------------
# REINFORCEMENT LEARNING CONFIGS
# ----------------------------------------------------------------------------
class RLConfig(BaseSchema):
    # General states and actions
    state_signals: Optional[List[str]] = None
    action_size: Optional[int] = None  # Number of possible actions
    history_length: int = 6  # Number of time steps to consider
    n_controllers: int = 1
    # Features
    add_hourly_time_features_to_state: bool = True
    add_daily_time_features_to_state: bool = True
    add_weekly_time_features_to_state: bool = True
    add_monthly_time_features_to_state: bool = True
    # Learning
    learning_rate: float = 0.001
    discount_factor: float = 1.0
    n_fit_epochs: int = 5
    batch_size: int = 32
    # Experience Replay
    experience_sampling_size: int = 1000
    replay_buffer_size: int = 20000
    prioritized_replay_alpha: float = 0.6
    prioritized_replay_beta0: float = 0.4
    prioritized_replay_eps: float = 1e-6


# ----------------------------------------------------------------------------
# AGENT CONFIGS
# ----------------------------------------------------------------------------
@unique
class SignalTags(str, Enum):
    STATE: str = "X"
    CONTROL: str = "U"
    EXOGENOUS: str = "W"
    OBSERVATION: str = "O"


class AgentDataConfig(BaseSchema):
    control_power_mapping: Dict[int, float]
    tracked_signals: List[str]
    signals_info: Dict[str, Any]


class AgentPredictionConfig(BaseSchema):
    # Signal inputs (k) with their history length (v) (0 = t)
    signal_inputs: Dict[str, int]
    signal_outputs: List[str]  # Signals output to predict
    ref_model_id: str  # Reference to use for the model registry


class AgentControlConfig(BaseSchema):
    trajectory_length: int = 6
    n_trajectory_samples: int = 1
    reinforcement_learning: RLConfig


class AgentConfig(BaseSchema):
    control: AgentControlConfig
    data: AgentDataConfig
    prediction: AgentPredictionConfig
    product: str = "demand_response"
    tariff: str = "flat_rate"


# ----------------------------------------------------------------------------
# SIMULATION CONFIGS
# ----------------------------------------------------------------------------
class SimulationTimeConfig(BaseSchema):
    start_time: str = "2023-01-01 00:00:00"
    end_time: str = "2023-01-07 00:00:00"
    step_size_s: int = 300


class SimulationGeographicalConfig(BaseSchema):
    location_lat: float
    location_lon: float
    location_alt: float = 0.0


# TODO: Add future asset configurations here
AssetConfigTypes = EnergyStorageConfig | BuildingConfig | ElectricVehicleConfig


class SimulationGlobalConfig(BaseSchema):
    directory: str = "simulation_results"
    time: SimulationTimeConfig
    geography: SimulationGeographicalConfig
    assets: dict[UidType, AssetConfigTypes]
    agent_config: AgentConfig
