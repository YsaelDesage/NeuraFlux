from neuraflux.schemas.config import (
    AgentConfig,
    AgentControlConfig,
    AgentDataConfig,
    AgentPredictionConfig,
    EnergyStorageConfig,
    RLConfig,
    SignalTags,
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
        "internal_energy": {
            "initial_value": 0,
            "tags": [SignalTags.STATE.value],
            "temporal_knowledge": 0,
        },
        "price": {
            "initial_value": 0,
            "tags": [SignalTags.EXOGENOUS.value],
            "temporal_knowledge": 0,
        },
    }
    CONTROL_POWER_MAPPING = {0: -250, 1: 0, 2: 250}
    INITIAL_STATE_DICT = {
        k: v["initial_value"]
        for k, v in SIGNALS_INFO.items()
        if v["initial_value"] is not None
    }
    ASSET_CONFIG = EnergyStorageConfig(
        control_power_mapping=CONTROL_POWER_MAPPING,
        capacity_kwh=500,
        initial_state_dict=INITIAL_STATE_DICT,
    )

    AGENT_CONFIG = AgentConfig(
        control=AgentControlConfig(
            n_trajectory_samples=2,
            trajectory_length=12,
            reinforcement_learning=RLConfig(
                state_signals=[
                    k
                    for k, v in SIGNALS_INFO.items()
                    if (
                        SignalTags.STATE in v["tags"]
                        or SignalTags.EXOGENOUS in v["tags"]
                    )
                ],
                action_size=len(CONTROL_POWER_MAPPING),
                learning_rate=5e-3,
                n_fit_epochs=5,
                experience_sampling_size=500,
            ),
        ),
        data=AgentDataConfig(
            control_power_mapping=CONTROL_POWER_MAPPING,
            tracked_signals=[
                k for k, v in SIGNALS_INFO.items() if SignalTags.STATE in v["tags"]
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
        product=AvailableProductsEnum.HOEP_MARKET.name.lower(),
        tariff=AvailableTariffsEnum.HOEP_MARKET.name.lower(),
    )

    SIMULATION_CONFIG = SimulationGlobalConfig(
        directory="sim_energy_storage_3_NEW",
        time=TIME_CONFIG,
        geography=GEO_CONFIG,
        assets={"1": ASSET_CONFIG},
        agent_config=AGENT_CONFIG,
    )
    # print(SIMULATION_CONFIG.model_dump_json(indent=4))

    simulation = Simulation(SIMULATION_CONFIG)
    simulation.run()
