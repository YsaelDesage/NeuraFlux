from neuraflux.local_typing import (
    DBSignalType,
    DBScalingDictsType,
    UidType,
    IndexType,
    DBTrajectoriesType,
)
from neuraflux.agency.products import Product
from neuraflux.agency.tariffs import Tariff
from neuraflux.agency.module import Module
from neuraflux.global_variables import (
    CONTROL_KEY,
    ENERGY_KEY,
    POWER_KEY,
    OAT_KEY,
)
import pandas as pd
import datetime as dt
import numpy as np

from neuraflux.agency.data.scaling_utils import (
    update_scaling_dict_from_df,
    scale_df_based_on_scaling_dict,
)
from neuraflux.agency.data.time_features import tf_all_cyclic
from neuraflux.weather_ref import WeatherRef


class DataModule(Module):
    """
    A class for managing data storage and retrieval during execution.

    Attributes:
    -----------
    db_asset_signals : dict
        A dictionary containing asset signal data.
    db_products : dict
        A dictionary containing agent products data.
    db_tariffs : dict
        A dictionary containing agent tariff data.
    db_scaling_dicts : dict
        A dictionary containing scaling values data.
    db_trajectories : dict
        A dictionary containing simulation trajectories data.
    """

    # ----------------------------------------------------------------------
    # SIGNAL DATA
    # ----------------------------------------------------------------------
    def get_augmented_history(
        self,
        uid: UidType,
        controls_power_mapping: dict[int, float],
        weather_ref: WeatherRef,
        scaled: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Get the augmented history of the asset.

        Args:
            uid (int): The unique identifier of the asset.
            power_mapping (Dict[int, float]): A dictionary mapping
                                              the controls to power.

        Returns:
            pd.DataFrame: The augmented history of the asset.
        """

        # Get pure asset signals
        df = self.get_asset_signal_history(uid)

        # Augment with all available data
        df = self.augment_df_with_all(
            uid, df, controls_power_mapping, weather_ref, **kwargs
        )
        if scaled:
            df = self.scale_data(uid, df)
        return df

    def augment_df_with_all(
        self,
        uid: UidType,
        df: pd.DataFrame,
        controls_power_mapping: dict[int, float],
        weather_ref: WeatherRef,
        time_features: bool = True,
    ) -> pd.DataFrame:
        """Augment the dataframe with all the possible data."""
        df = df.copy()

        # Weather
        df = self.augment_dataframe_with_weather(df, weather_ref)

        # Virtual metering
        df = self.augment_dataframe_with_virtual_metering_data(
            df, controls_power_mapping
        )

        # Tariff
        df = self.augment_dataframe_with_tariff(df, uid)

        # Products
        df = self.augment_dataframe_with_product_rewards(df, uid)

        # Time features
        df = tf_all_cyclic(df) if time_features else df

        return df

    def augment_dataframe_with_weather(
        self, df: pd.DataFrame, weather_ref
    ) -> pd.DataFrame:
        """Augment the dataframe with weather information.

        Args:
            df (pd.DataFrame): The dataframe to augment.

        Returns:
            pd.DataFrame: The augmented dataframe.
        """
        for idx in df.index:
            df.loc[
                idx, "outside_air_temperature"
            ] = weather_ref.get_temperature_at_time(idx)
        return df

    def push_asset_signal_data(
        self,
        uid: UidType,
        index: IndexType,
        signals_dict: dict[str, float | int | str],
    ) -> None:
        """Pushes a dictionary of signals to the asset's historical data.

        Args:
            uid (Union[int, str]): The unique identifier of the asset.
            index (Union[int, dt.datetime]): The index of the data.
            signals_dict (Dict[str, Union[float, int, str]]): A dictionary of
                                                              signals.
        """
        if uid not in self.db_asset_signals:
            self.db_asset_signals[uid] = {}

        # If the index already exists, update the data, else assign to it
        if index in self.db_asset_signals[uid].keys():
            self.db_asset_signals[uid][index].update(signals_dict)
        else:
            self.db_asset_signals[uid][index] = signals_dict
        return None

    def get_asset_signal_history(self, uid: UidType) -> pd.DataFrame:
        data = self.db_asset_signals[uid]
        df = pd.DataFrame.from_dict(data, orient="index")

        # Convert control columns to type Int64
        control_cols = [
            col for col in df.columns if col.startswith(CONTROL_KEY)
        ]
        if control_cols:
            for col in control_cols:
                df[col] = df[col].astype("Int64")

        return df

    # ----------------------------------------------------------------------
    # VIRTUAL METER DATA
    # ----------------------------------------------------------------------
    def augment_dataframe_with_virtual_metering_data(
        self, df: pd.DataFrame, control_power_mapping: dict[int, float]
    ) -> pd.DataFrame:
        """Calculate the virtual metering data for the specified asset."""

        df = df.copy()
        n_controls = sum([col.startswith(CONTROL_KEY) for col in df.columns])

        # Power calculation
        if len(df) > 0 and n_controls > 0:
            # Identify control columns
            control_columns = [
                col for col in df.columns if col.startswith(CONTROL_KEY)
            ]

            # Map each control column to its power and sum them
            df[POWER_KEY] = (
                df[control_columns]
                .apply(lambda x: x.map(control_power_mapping))
                .sum(axis=1)
            )

            # Energy calculation
            if isinstance(df.index, pd.DatetimeIndex):
                # Use an average delta value for missing deltas
                avg_delta = df.index.to_series().diff().mean()

                # If the delta does not exists (likely single row), no fill
                if avg_delta is pd.NaT:
                    time_diff = df.index.to_series().diff()
                # Else use average delta as value
                elif isinstance(avg_delta, dt.timedelta):
                    time_diff = (
                        df.index.to_series()
                        .diff()
                        .fillna(
                            pd.Timedelta(seconds=avg_delta.total_seconds())
                        )
                    )
                else:
                    raise ValueError(
                        f"Unknown avg_delta type: {type(avg_delta)}"
                    )
                energy = df[POWER_KEY] * time_diff.dt.seconds / 3600
            elif isinstance(df.index, int):
                time_diff = df.index.to_series().diff().fillna(0)
                energy = df[POWER_KEY] * time_diff
            else:
                raise ValueError("Unsupported index type.")

            df[ENERGY_KEY] = energy
        else:
            df[ENERGY_KEY] = np.nan
            df[POWER_KEY] = np.nan

        return df

    # ----------------------------------------------------------------------
    # SCALING
    # ----------------------------------------------------------------------
    def fit_scaling_dict_from_df(
        self, uid: UidType, df: pd.DataFrame, columns_to_scale: list[str] = []
    ) -> None:
        # Get the existing min/max dictionary for the asset, if present
        scaling_dict = (
            self.db_scaling_dicts[uid] if uid in self.db_scaling_dicts else {}
        )

        # Update (existing) scaling dict with min/max values from the dataframe
        scaling_dict = update_scaling_dict_from_df(
            df, scaling_dict, columns_to_scale
        )

        # Update the scaling dict in the database
        self.db_scaling_dicts[uid] = scaling_dict

    def update_scaling_dict(
        self, uid: UidType, new_scaling_dict: dict[str, tuple[float, float]]
    ) -> None:
        # Get the existing min/max dictionary for the asset, if present
        scaling_dict = (
            self.db_scaling_dicts[uid] if uid in self.db_scaling_dicts else {}
        )
        # Update (existing) scaling dict with entries from the new one
        scaling_dict.update(new_scaling_dict)

        # Update the scaling dict in the database
        self.db_scaling_dicts[uid] = scaling_dict

    def scale_data(self, uid: UidType, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if uid not in self.db_scaling_dicts:
            print("Warning: unable to scale data, no scaling dict found.")
            return df

        scaling_dict = self.db_scaling_dicts[uid]
        df = scale_df_based_on_scaling_dict(df, scaling_dict)
        return df

    # ----------------------------------------------------------------------
    # TARIFF AND PRICING DATA
    # ----------------------------------------------------------------------
    def assign_tariff_structure(self, uid: UidType, tariff: Tariff) -> None:
        """Assign a tariff to the specified asset.

        Args:
            tariff (Tariff): The tariff object to assign to this instance.
        """
        self.db_tariffs[uid] = tariff

    def augment_dataframe_with_tariff(
        self, df: pd.DataFrame, uid: UidType
    ) -> pd.DataFrame:
        """Augment the dataframe with tariff information.

        Args:
            df (pd.DataFrame): The dataframe to augment.
            uid (Union[int, str]): The unique identifier of the agent.

        Returns:
            pd.DataFrame: The augmented dataframe.
        """
        tariff = self.db_tariffs[uid]
        df = tariff.calculate_price_vector(df.copy())
        return df

    # ----------------------------------------------------------------------
    # PRODUCT DATA
    # ----------------------------------------------------------------------
    def assign_product(self, uid: UidType, product: Product) -> None:
        """Adds a product to the specified asset.

        Args:
            product (Product): The product object to add to this instance.
        """
        if uid not in self.db_products:
            self.db_products[uid] = []
        self.db_products[uid].append(product)

    def get_products(self, uid: UidType) -> list[Product]:
        if uid in self.db_products:
            return self.db_products[uid]
        return []

    def remove_product(self, uid: UidType, product: Product) -> None:
        """Removes a specific product from the asset.

        Args:
            uid (Union[int, str]): The unique identifier of the agent.
            product (Product): The product object to remove from this instance.
        """
        for product in self.db_products[uid]:
            if isinstance(product, product.__class__):
                self.db_products[uid].remove(product)

    def augment_dataframe_with_product_rewards(
        self, df: pd.DataFrame, uid: UidType
    ) -> pd.DataFrame:
        """Augment the dataframe with product reward information.

        Args:
            df (pd.DataFrame): The dataframe to augment.
            uid (Union[int, str]): The unique identifier of the agent.

        Returns:
            pd.DataFrame: The augmented dataframe.
        """
        products = self.get_products(uid)
        for product in products:
            reward_names = product.get_reward_names()
            if len(reward_names) == 1:
                reward_names = reward_names[0]
            df[reward_names] = product.calculate_rewards(df)

            # Add dones 
            df = product.calculate_dones(df)

            # Add additional product-defined features to data
            df = product.add_features(df)
        return df

    def get_product_rewards(self, uid: UidType) -> list[str] | None:
        """Returns the list of product rewards for the specified asset.

        Args:
            uid (Union[int, str]): The unique identifier of the agent.

        Returns:
            List[str] | None: The list of product rewards.
        """
        products = self.get_products(uid)
        rewards = []
        for product in products:
            rewards = rewards + product.get_reward_names()
        return rewards

    # ----------------------------------------------------------------------
    # TRAJECTORY DATA
    # ----------------------------------------------------------------------
    def push_trajectory_data(
        self,
        uid: UidType,
        index: IndexType,
        trajectory: pd.DataFrame,
    ) -> None:
        """Pushes a dataframe of simulated data from an asset observation.

        Args:
            uid (Union[int, str]): The unique identifier of the asset.
            index (Union[int, dt.datetime]): The index of the data.
            signals_dict (Dict[str, Union[float, int, str]]): A dictionary of
                                                              signals.
        """
        if uid not in self.db_trajectories:
            self.db_trajectories[uid] = {}

        # If the index already exists, update the data, else assign to it
        if index in self.db_trajectories[uid].keys():
            self.db_trajectories[uid][index].append(trajectory)
        else:
            self.db_trajectories[uid][index] = [trajectory]
        return None

    def get_trajectory_data(
        self, uid: UidType, index: IndexType
    ) -> list[pd.DataFrame] | None:
        """Returns the simulated trajectory data for the specified asset,
        for a given index.

        Args:
            uid (Union[int, str]): The unique identifier of the asset.
            index (IndexType): The index of the data.

        Returns:
            List[pd.DataFrame] | None: The simulated trajectory data.
        """
        # If the index is present, return data, else return None
        if uid in self.db_trajectories.keys():
            data = self.db_trajectories[uid]
            if index in data.keys():
                return data[index]
        return None

    # ----------------------------------------------------------------------
    # PRIVATE METHODS
    # ----------------------------------------------------------------------
    def _initialize_data_structures(self):
        # Assets signal data storage
        self.db_asset_signals: DBSignalType = {}

        # Agents products data storage
        self.db_products: dict[UidType, list[Product]] = {}

        # Agent tariff data storage
        self.db_tariffs: dict[UidType, Tariff] = {}

        # Scaling values data storage
        self.db_scaling_dicts: DBScalingDictsType = {}

        # Simulation trajectories data storage
        self.db_trajectories: DBTrajectoriesType = {}
