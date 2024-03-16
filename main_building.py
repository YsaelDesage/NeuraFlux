from neuraflux.schemas.config import (
    AgentConfig,
    AgentControlConfig,
    AgentDataConfig,
    AgentPredictionConfig,
    BuildingConfig,
    SignalTags,
    RLConfig,
    SimulationGeographicalConfig,
    SimulationGlobalConfig,
    SimulationTimeConfig,
)
from neuraflux.agency.products import AvailableProductsEnum
from neuraflux.agency.tariffs import AvailableTariffsEnum
from neuraflux.simulation import Simulation

if __name__ == "__main__":
    TIME_CONFIG = SimulationTimeConfig(
        start_time="2023-01-01_00-00-00",
        end_time="2023-01-12_00-00-00",
        step_size_s=300,
    )
    GEO_CONFIG = SimulationGeographicalConfig(
        location_lat=43.6532, location_lon=-79.3832, location_alt=76
    )
    SIGNALS_INFO = {
        "temperature_1": {
            "initial_value": 21.0,
            "tags": [SignalTags.STATE.value],
            "temporal_knowledge": 0,
        },
        "temperature_2": {
            "initial_value": 21.0,
            "tags": [SignalTags.STATE.value],
            "temporal_knowledge": 0,
        },
        "temperature_3": {
            "initial_value": 21.0,
            "tags": [SignalTags.STATE.value],
            "temporal_knowledge": 0,
        },
        "hvac_1": {
            "initial_value": 0,
            "tags": [SignalTags.CONTROL.value],
            "temporal_knowledge": 0,
        },
        "hvac_2": {
            "initial_value": 0,
            "tags": [SignalTags.CONTROL.value],
            "temporal_knowledge": 0,
        },
        "hvac_3": {
            "initial_value": 0,
            "tags": [SignalTags.CONTROL.value],
            "temporal_knowledge": 0,
        },
        "cool_setpoint": {
            "initial_value": 24.0,
            "tags": [SignalTags.EXOGENOUS.value],
            "temporal_knowledge": 0,
        },
        "heat_setpoint": {
            "initial_value": 18.0,
            "tags": [SignalTags.EXOGENOUS.value],
            "temporal_knowledge": 0,
        },
        "occupancy": {
            "initial_value": 0,
            "tags": [SignalTags.EXOGENOUS.value],
            "temporal_knowledge": 0,
        },
        "outside_air_temperature": {
            "initial_value": None,
            "tags": [SignalTags.EXOGENOUS.value],
            "temporal_knowledge": 0,
        },
    }
    CONTROL_POWER_MAPPING = {
        0: 20,
        1: 10,
        2: 0,
        3: 10,
        4: 20,
    }
    INITIAL_STATE_DICT = {
        "temperature": [21.0, 21.0, 21.0],
        "hvac": [0, 0, 0],
    }
    ASSET_CONFIG = BuildingConfig(
        control_power_mapping=CONTROL_POWER_MAPPING,
        initial_state_dict=INITIAL_STATE_DICT,
        n_controls=3,
    )

    AGENT_CONFIG = AgentConfig(
        control=AgentControlConfig(
            n_trajectory_samples=1,
            trajectory_length=12,
            reinforcement_learning=RLConfig(
                state_signals=[
                    "temperature_1",
                    "temperature_2",
                    "temperature_3",
                    "cool_setpoint",
                    "heat_setpoint",
                    "occupancy",
                    "outside_air_temperature",
                ],
                action_size=len(CONTROL_POWER_MAPPING),
                n_controllers=3,
                learning_rate=5e-3,
                n_fit_epochs=5,
                experience_sampling_size=500,
            ),
        ),
        data=AgentDataConfig(
            control_power_mapping=CONTROL_POWER_MAPPING,
            tracked_signals=[
                "temperature",
                "hvac",
                "cool_setpoint",
                "heat_setpoint",
                "occupancy",
            ],
            signals_info=SIGNALS_INFO,
        ),
        prediction=AgentPredictionConfig(
            signal_inputs={
                k: v["temporal_knowledge"]
                for k, v in SIGNALS_INFO.items()
                if v["temporal_knowledge"] is not None
            },
            signal_outputs=[
                k for k, v in SIGNALS_INFO.items() if SignalTags.STATE in v["tags"]
            ],
            ref_model_id="12345",
        ),
        product=AvailableProductsEnum.HVAC_BUILDING.name.lower(),
        tariff=AvailableTariffsEnum.FLAT_RATE.name.lower(),
    )

    SIMULATION_CONFIG = SimulationGlobalConfig(
        directory="sim_building_2h_NEW_STRICT_DISCOMFORT",
        time=TIME_CONFIG,
        geography=GEO_CONFIG,
        assets={"1": ASSET_CONFIG},
        agent_config=AGENT_CONFIG,
    )
    # print(SIMULATION_CONFIG.model_dump_json(indent=4))

    simulation = Simulation(SIMULATION_CONFIG)
    simulation.run()
