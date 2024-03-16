from abc import ABCMeta, abstractmethod
from enum import Enum, unique
import pandas as pd
import numpy as np
import datetime as dt

from neuraflux.global_variables import ENERGY_KEY, REWARD_KEY, DONE_KEY, TARIFF_KEY
from pandas.core.api import DataFrame as DataFrame
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt


class Product(metaclass=ABCMeta):
    """Base class for all assets."""

    def __init__(self, period: int = 24):
        if period > 24:
            raise ValueError("Product period cannot exceed 24h")
        self.period = period

    @abstractmethod
    def calculate_rewards(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_reward_names(self) -> list[str]:
        raise NotImplementedError

    def calculate_dones(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Ensure the index is a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")

        # Calculate the minute of the day for each row
        minutes_of_day = df.index.hour * 60 + df.index.minute

        # Check if the minute of the day is 5 minutes before the end of a cycle
        cycle_minutes = self.period * 60
        df[DONE_KEY] = (((minutes_of_day + 5) % cycle_minutes) == 0).astype(bool)

        return df

    # For products offering additional features in learning process
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Return original df with no features, by default
        return df


class DemandResponseProduct(Product):
    def calculate_rewards(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        df[REWARD_KEY] = -df[ENERGY_KEY].values  # type: ignore
        df.loc[
            (df.index.hour < 15) | (df.index.hour >= 19), REWARD_KEY  # type: ignore # noqa: E501
        ] = 0
        return df[[REWARD_KEY]].values

    def get_reward_names(self) -> list[str]:
        return [REWARD_KEY + "_DR"]


class CAISODynamicPricingProduct(Product):
    def __init__(
        self,
        price_file_path: str = "neuraflux/agency/datasets/dynamic_pricing_2023.csv",
    ):
        # Reading the dynamic prices from the provided CSV file
        self.dynamic_prices_df = pd.read_csv(
            price_file_path, index_col=0, parse_dates=True
        )

    def get_price(self, time: dt.datetime) -> float:
        # Find the closest time index in the DataFrame to the given time
        closest_time = self.dynamic_prices_df.index.get_loc(time)
        return self.dynamic_prices_df.iloc[closest_time]["price"]

    def calculate_rewards(self, df: pd.DataFrame) -> pd.DataFrame:
        # Assuming the DataFrame has a datetime index and an 'energy' column
        df = df.copy()
        prices = [
            self.get_price(time=row.name) * -row[ENERGY_KEY]
            for index, row in df.iterrows()
        ]
        df[REWARD_KEY + "_ENERGY"] = prices
        return df

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Copy the DataFrame to avoid modifying the original one
        df = df.copy()

        # Iterate over the DataFrame rows
        for index, _ in df.iterrows():
            # Calculate the timestamps for the next 12 intervals (5 minutes each)
            next_times = [index + dt.timedelta(minutes=5 * i) for i in range(0, 13)]

            # Retrieve prices for these timestamps
            for i, time in enumerate(next_times):
                # Check if the time exists in dynamic_prices_df, if not, handle appropriately
                if time in self.dynamic_prices_df.index:
                    price = self.get_price(time)
                else:
                    # Handle missing price data (e.g., set to NaN or use some default value)
                    price = float("nan")  # or some other handling strategy

                # Add the price to the DataFrame
                df.loc[index, f"price_t+{i}"] = price

        return df

    def get_reward_names(self) -> list[str]:
        reward_names = [REWARD_KEY + "_ENERGY"]
        return reward_names


class HOEPMarketProduct(Product):
    def __init__(
        self,
        price_file_path: str = "neuraflux/agency/datasets/hoep_interpolated_2023.csv",
        period: int = 3,
    ):
        # Reading the dynamic prices from the provided CSV file
        self.dynamic_prices_df = pd.read_csv(
            price_file_path, index_col=0, parse_dates=True
        )
        self.dynamic_prices_df.index = self.dynamic_prices_df.index - dt.timedelta(
            hours=1
        )

        self.period = period

    def get_price(self, time: dt.datetime) -> float:
        # Find the closest time index in the DataFrame to the given time
        closest_time = self.dynamic_prices_df.index.get_loc(time)
        return self.dynamic_prices_df.iloc[closest_time]["HOEP"]

    def calculate_rewards(self, df: DataFrame) -> np.ndarray:
        # Assuming the DataFrame has a datetime index and an 'energy' column
        df = df.copy()
        prices = [
            self.get_price(time=row.name) * -row[ENERGY_KEY]
            for index, row in df.iterrows()
        ]
        df[REWARD_KEY + "_ENERGY"] = prices

        # Set the value to be negative penalty if internal energy is 0 and action is 0
        if "control_1" in df.columns:
            filter = (df["internal_energy"] == 0) & (df["control_1"] == 0)
            df.loc[filter, REWARD_KEY + "_ENERGY"] = -2500

        return df[REWARD_KEY + "_ENERGY"].values

    def get_reward_names(self) -> list[str]:
        reward_names = [REWARD_KEY + "_ENERGY"]
        return reward_names


class HVACBuildingProduct(Product):
    def __init__(self, period: int = 2):
        self.period = period

    def add_features(self, df):
        df = df.copy()
        if "control_1" in df.columns:
            df[["hvac_1", "hvac_2", "hvac_3"]] = (
                df[["control_1", "control_2", "control_3"]] - 2
            )
        return df

    def calculate_rewards(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()

        # Calculate comfort reward as the absolute deviation from
        # cooling setpoint if warmer, or heating setpoint if colder
        # 0 otherwise
        discomforts = []
        for zone in range(1, 4):
            discomfort = np.zeros(df.shape[0])

            filter1 = (df[f"temperature_{zone}"] > df["cool_setpoint"]).values
            discomfort[filter1] = (
                df["cool_setpoint"] - df[f"temperature_{zone}"]
            ).values[filter1]

            filter2 = (df[f"temperature_{zone}"] < df["heat_setpoint"]).values
            discomfort[filter2] = (
                df[f"temperature_{zone}"] - df["heat_setpoint"]
            ).values[filter2]
            discomforts.append(discomfort)

        df[REWARD_KEY + "_DISCOMFORT"] = (
            discomforts[0] + discomforts[1] + discomforts[2]
        )

        # Create a filter where all control keys are equal to 2, and where discomfort is 0
        if "control_1" in df.columns:
            filter = (
                (df["control_1"] == 2)
                & (df["control_2"] == 2)
                & (df["control_3"] == 2)
                & (df[REWARD_KEY + "_DISCOMFORT"] == 0)
            )

            df.loc[filter, REWARD_KEY + "_DISCOMFORT"] = 1

        # Calculate energy cost reward (M Rate HydroQuebec)
        df[REWARD_KEY + "_COST"] = df[ENERGY_KEY].values * -0.005567

        # Calculate energy reward as the energy consumption
        df[REWARD_KEY + "_ENERGY"] = df[ENERGY_KEY].values * -1

        return df[self.get_reward_names()].values

    def get_reward_names(self) -> list[str]:
        reward_names = [
            REWARD_KEY + "_DISCOMFORT",
            REWARD_KEY + "_COST",
            REWARD_KEY + "_ENERGY",
        ]
        return reward_names


class EvDrTouGhgProduct(Product):
    def __init__(self, period: int = 2):
        self.period = period
        self.duck_curve_values = [
            0.2,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.55,
            0.7,
            0.55,
            0.5,
            0.2,
            0.1,
            0.0,
            0.1,
            0.2,
            0.4,
            0.5,
            0.7,
            0.9,
            1.0,
            0.9,
            0.75,
            0.5,
            0.3,
        ]

    def get_emission_rate(self, time: dt.datetime) -> float:
        hour = time.hour
        seed = time.day

        np.random.seed(seed)  # Set the seed for reproducibility

        # Get the base value from the hardcoded array
        base_value = self.duck_curve_values[hour]

        # Add random variation to the value, e.g., +/- 5%
        variation = np.random.uniform(-0.1, 0.1)
        final_value = np.clip(base_value + variation, 0, 1)

        return final_value

    def is_dr_event(self, time: dt.datetime) -> bool:
        """
        Determine if the given time falls within a DR event.

        :param time: A datetime object representing the current time.
        :return: A boolean indicating if the time is within a DR event.
        """
        # Use the day of the year as the seed
        seed = time.day
        np.random.seed(seed)

        # Randomly select an hour for the DR event between 5 PM and 8 PM
        dr_event_hour = np.random.choice([17, 18, 19])  # 5 PM, 6 PM, or 7 PM

        # Check if the current hour matches the DR event hour
        return time.hour == dr_event_hour

    def calculate_rewards(self, df: pd.DataFrame) -> pd.DataFrame:
        # Assuming the DataFrame has a datetime index and an 'energy' column
        df = df.copy()
        emissions = [
            self.get_emission_rate(time=row.name) * -row[ENERGY_KEY]
            for index, row in df.iterrows()
        ]
        cost = [
            min((row[TARIFF_KEY] * -row[ENERGY_KEY]), 0)
            + (self.is_dr_event(row.name) * -row[ENERGY_KEY])
            for index, row in df.iterrows()
        ]
        df[REWARD_KEY + "_COST"] = cost
        df[REWARD_KEY + "_GHG"] = emissions
        return df[self.get_reward_names()].values

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Copy the DataFrame to avoid modifying the original one
        df = df.copy()

        # Add event column
        dr_event = [self.is_dr_event(time=row.name) for index, row in df.iterrows()]
        df["dr_event"] = dr_event

        # Add GHG emission rate column
        emissions = [
            self.get_emission_rate(time=row.name) for index, row in df.iterrows()
        ]
        df["emission_rate"] = emissions

        return df

    def get_reward_names(self) -> list[str]:
        reward_names = [REWARD_KEY + "_COST", REWARD_KEY + "_GHG"]
        return reward_names


@unique
class AvailableProductsEnum(Enum):
    DEMAND_RESPONSE = DemandResponseProduct
    HVAC_BUILDING = HVACBuildingProduct
    HOEP_MARKET = HOEPMarketProduct
    DYNAMIC_PRICING = CAISODynamicPricingProduct
    EV_DR = EvDrTouGhgProduct

    @classmethod
    def list_products(cls):
        return list(map(lambda product: product.name.lower(), cls))

    @classmethod
    def get_product_class(cls, product_name: str):
        for product in cls:
            if product.name.lower() == product_name.lower():
                return product.value
        raise ValueError(f"No product found with name: {product_name}")


class ProductFactory:
    @staticmethod
    def create(product_name: str, **kwargs) -> Product:
        product_class = AvailableProductsEnum.get_product_class(product_name)
        return product_class(**kwargs)
