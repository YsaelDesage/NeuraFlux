from neuraflux.agency.module import Module
from neuraflux.local_typing import UidType
from neuraflux.schemas.config import AgentConfig


# TODO: Add validation process when pushing new configs
class ConfigModule(Module):
    # -----------------------------------------------------------------------
    # AGENT
    # -----------------------------------------------------------------------
    def push_new_agent_config(
        self, uid: str | int, agent_config: AgentConfig
    ) -> None:
        self.db_agent_configs[uid] = agent_config

    def get_agent_config(self, uid: str | int) -> AgentConfig:
        return self.db_agent_configs[uid]

    # -----------------------------------------------------------------------
    # INTERNAL
    # -----------------------------------------------------------------------
    def _initialize_data_structures(self) -> None:
        self.db_agent_configs: dict[UidType, AgentConfig] = {}
