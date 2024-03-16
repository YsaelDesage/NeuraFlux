import datetime as dt
import dill
from abc import ABCMeta, abstractmethod
from copy import copy
from os.path import join, exists
from typing import Any, Optional, Union
import numpy as np

import pandas as pd

from neuraflux.global_variables import CONTROL_KEY, TIMESTAMP_KEY
from neuraflux.schemas.control import DiscreteControl


class Asset(metaclass=ABCMeta):
    """Base class for all assets."""

    def __init__(
        self,
        name: str,
        config: Any,
        timestamp: dt.datetime,
        outside_air_temperature: float,
    ) -> None:
        self.name = name
        self.config = config

        # Initialize internal simulation variables
        self.timestamp: dt.datetime = timestamp
        self.outside_air_temperature: float = outside_air_temperature
        self.control: list[int] = []
        self.power: Optional[float] = None

        self.initialized: bool = False

        # Initialize history for variables tracked in the simulation
        tracked_variables = (
            self.config.tracked_variables + self.config.core_variables
        )

        # Handle control separately, as it is a list of unknown size
        tracked_variables.remove(CONTROL_KEY)
        tracked_variables += [
            CONTROL_KEY + "_" + str(i + 1)
            for i in range(self.config.n_controls)
        ]

        # Define history dict, expanding lists into multiple variables
        self.history = {}
        init_variables = self.config.initial_state_dict
        for variable in tracked_variables:
            if variable in init_variables and isinstance(
                init_variables[variable], (list, tuple, np.ndarray)
            ):
                for i in range(len(init_variables[variable])):
                    self.history[variable + "_" + str(i + 1)] = []
            else:
                self.history[variable] = []

        # Initialize internal variables using initial state
        if self.config.initial_state_dict:
            self.initialized = True
            for key, value in self.config.initial_state_dict.items():
                setattr(self, key, value)

        self._update_tracked_variables()

    @abstractmethod
    def step(
        self,
        control: list[DiscreteControl],
        timestamp: int | dt.datetime,
        outside_air_temperature: float,
    ) -> Optional[float]:
        """Perform a step in the simulation, given a submitted control.
        This function must return the power consumption of the asset at
        this step step. Positive values indicate power consumption, while
        negative values indicate power generation.

        Args:
            control (int): control to be performed.
            timestamp (Union[int, dt.datetime]): timestamp of the simulation.
            outside_air_temperature (float): outside air temperature.

        Returns:
            float: Power consumed by the asset.
        """
        if not self.initialized:
            raise ValueError("Asset must be initialized first.")
        self.timestamp = timestamp
        self.control = [c.value for c in control]
        self.outside_air_temperature = outside_air_temperature
        self._update_tracked_variables()

        # NOTE: The child class must define the power variable
        return self.power

    @abstractmethod
    def get_auto_control(
        self, timestamp: Union[int, dt.datetime]
    ) -> list[DiscreteControl]:
        """Get the automatic control for the asset.

        Args:
            timestamp (Union[int, dt.datetime]): timestamp of the simulation.

        Returns:
            list[DiscreteControl]: The automatic control.
        """
        raise NotImplementedError

    def auto_step(
        self,
        timestamp: Union[int, dt.datetime],
        outside_air_temperature: float | None,
    ) -> float:
        """Perform a step in the simulation, automatically choosing control.

        Args:
            timestamp (Union[int, dt.datetime]): timestamp of the simulation.
            outside_air_temperature (float): outside air temperature.

        Returns:
            float: Power consumed by the asset.
        """
        control = self.get_auto_control(timestamp, outside_air_temperature)
        return self.step(control, timestamp, outside_air_temperature)

    def augment_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def get_signal(self, signal: str) -> Any:
        if hasattr(self, signal):
            return getattr(self, signal)
        return np.nan

    def get_historical_data(self, nan_padding: bool = False) -> pd.DataFrame:
        """Returns a dataframe with all variables tracked in the simulation.

        Returns:
            pd.DataFrame: Dataframe with all variables tracked in simulation.
        """

        # Work with a copy of the data to avoid modifying the original
        data = copy(self.history)

        # Add None values to the end of the list to equalize length
        if nan_padding:
            max_length = max(len(value) for value in data.values())
            data_clean = {
                key: value + [None] * (max_length - len(value))
                for key, value in data.items()
            }

        # Remove last timestamps missing control and power data
        else:
            min_length = min(len(value) for value in data.values())
            data_clean = {
                key: value[:min_length] for key, value in data.items()
            }

        # Separate index column and clean data columns to create dataframe
        timestamp_data = data_clean.pop(TIMESTAMP_KEY)
        df = pd.DataFrame(data_clean, index=timestamp_data)
        return df

    def to_file(self, directory: str = "") -> None:
        """
        Save asset's state to a file using dill.

        Args:
            filename (str, optional): Name of the file to save.
            directory (str, optional): Save directory. Defaults to "".
        """
        filepath = join(directory, self.name + ".pkl")
        with open(filepath, "wb") as f:
            dill.dump(self, f)

    @classmethod
    def from_file(cls, name: str, directory: str = "") -> "Asset":
        """
        Load an asset's state from a file using dill.

        Args:
            filename (str): Name of the file to load from.
            directory (str, optional): Load directory. Defaults to "".
        """
        filepath = join(directory, name + ".pkl")
        with open(filepath, "rb") as f:
            return dill.load(f)

    @classmethod
    def load_or_initialize(
        cls,
        name: str,
        directory: str = "",
        config: Any = None,
        timestamp: dt.datetime | None = None,
        outside_air_temperature: float | None = None,
        **kwargs
    ) -> "Asset":
        """
        Load an asset's state from a file if it exists, or initialize a new
        instance.

        Args:
            filename (str): Name of the file to load from.
            directory (str, optional): Directory to check for the file.
            Defaults to "".
            **kwargs: Additional keyword arguments to pass to the initializer.

        Returns:
            Asset: A loaded or a new instance of Asset.
        """
        filepath = join(directory, name + ".pkl")
        if exists(filepath):
            with open(filepath, "rb") as f:
                return dill.load(f)
        elif config is not None and timestamp is not None:
            return cls(
                name, config, timestamp, outside_air_temperature, **kwargs
            )
        else:
            raise ValueError(
                "Asset does not exist and missing initialization variables."
            )

    def _update_tracked_variables(self) -> None:
        # Define all tracked variables to add to history
        tracked_variables = (
            self.config.tracked_variables + self.config.core_variables
        )

        # Add all other standard variables
        for variable_name in tracked_variables:
            variable = getattr(self, variable_name)
            if isinstance(variable, (list, tuple, np.ndarray)):
                for i, value in enumerate(variable):
                    self.history[variable_name + "_" + str(i + 1)].append(
                        value
                    )
            elif variable is not None:
                self.history[variable_name].append(variable)
