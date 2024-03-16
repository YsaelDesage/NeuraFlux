import datetime as dt
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional, Tuple
from enum import Enum, unique

import pandas as pd
import numpy as np

from neuraflux.global_variables import ENERGY_KEY, TARIFF_KEY


class Tariff(metaclass=ABCMeta):
    @abstractmethod
    def calculate_price(
        self,
        time: Optional[dt.datetime] = None,
        power: Optional[float] = None,
        energy: Optional[float] = None,
        other_info: Dict[str, Any] = {},
    ) -> Optional[float]:
        raise NotImplementedError

    @abstractmethod
    def calculate_price_vector(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class FlatRateTariff(Tariff):
    def __init__(self, rate: float = 0.1):
        self.rate = rate

    def calculate_price(
        self,
        time: Optional[dt.datetime] = None,
        power: Optional[float] = None,
        energy: Optional[float] = None,
        other_info: Dict[str, Any] = {},
    ) -> Optional[float]:
        if energy is not None:
            return self.rate * energy
        return None

    def calculate_price_vector(self, df: pd.DataFrame) -> pd.DataFrame:
        df[TARIFF_KEY] = df[ENERGY_KEY].values * self.rate  # type: ignore
        return df


class HydroQuebecDTariff(Tariff):
    def __init__(
        self,
        patrimonial_value: float = 0.06509,
        highest_value: float = 0.10041,
        cuttoff_energy_value: float = 40.0,
    ):
        self.patrimonial_value = patrimonial_value
        self.highest_value = highest_value
        self.cuttoff_energy_value = cuttoff_energy_value

    def calculate_price(
        self,
        time: Optional[dt.datetime] = None,
        power: Optional[float] = None,
        energy: Optional[float] = None,
        other_info: Dict[str, Any] = {},
    ) -> Optional[float]:
        if energy is not None:
            period_energy = (
                0
                if "period_energy" not in other_info.keys()
                else other_info["period_energy"]
            )
            low_consumption_energy = min(self.cuttoff_energy_value, period_energy)
            high_consumption_energy = max(energy - self.cuttoff_energy_value, 0)
            price = float(
                low_consumption_energy * self.patrimonial_value
                + high_consumption_energy * self.highest_value
            )
            return price
        return None

    @abstractmethod
    def calculate_price_vector(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class HydroQuebecMTariff(Tariff):
    def __init__(
        self,
        energy_price_kwh: float = 0.005567,
        power_price_kw: float = 16.139,
        cuttoff_energy_value: int = 210000,
    ):
        self.energy_price_kwh = energy_price_kwh
        self.power_price_kw = power_price_kw
        self.cuttoff_energy_value = cuttoff_energy_value

    def calculate_price(
        self,
        time: Optional[dt.datetime] = None,
        power: Optional[float] = None,
        energy: Optional[float] = None,
        other_info: Dict[str, Any] = {},
    ) -> Optional[float]:
        if energy is not None:
            return self.energy_price_kwh * energy
        return None

    @abstractmethod
    def calculate_price_vector(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class OntarioTOUTariff(Tariff):
    def __init__(
        self,
        on_peak: float = 0.151,
        mid_peak: float = 0.102,
        off_peak: float = 0.074,
        on_peak_range: Tuple[int, int] = (17, 19),
        mid_peak_ranges: Tuple[Tuple[int, int], Tuple[int, int]] = (
            (7, 11),
            (17, 19),
        ),
    ):
        self.on_peak = on_peak
        self.mid_peak = mid_peak
        self.off_peak = off_peak
        self.on_peak_range = on_peak_range
        self.mid_peak_ranges = mid_peak_ranges

    def calculate_price(
        self,
        time: Optional[dt.datetime] = None,
        power: Optional[float] = None,
        energy: Optional[float] = None,
        other_info: Dict[str, Any] = {},
    ) -> Optional[float]:
        if time is not None and energy is not None:
            hour = time.hour

            # Apply Ontario's TOU based on 24h
            if self.on_peak_range[1] > hour >= self.on_peak_range[0]:
                price = self.on_peak * energy / 1000  #
            if (
                self.mid_peak_ranges[0][1] > hour >= self.mid_peak_ranges[0][0]
                or self.mid_peak_ranges[1][1] > hour >= self.mid_peak_ranges[1][0]
            ):
                price = self.mid_peak * energy / 1000
            price = self.off_peak * energy / 1000

            # Not paid injecting energy back on the grid
            return max(price, 0)
        return None

    def calculate_price_vector(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for index, row in df.iterrows():
            df.loc[index, TARIFF_KEY] = self.calculate_price(
                time=row.name, energy=row[ENERGY_KEY]
            )
        return df


class DynamicPricingTariff(Tariff):
    def __init__(
        self,
        price_file_path: str = "neuraflux/agency/datasets/dynamic_pricing_2023.csv",
    ):
        # Reading the dynamic prices from the provided CSV file
        self.dynamic_prices_df = pd.read_csv(
            price_file_path, index_col=0, parse_dates=True
        )

    def calculate_price(
        self,
        time: Optional[dt.datetime] = None,
        power: Optional[float] = None,
        energy: Optional[float] = None,
        other_info: Dict[str, Any] = {},
    ) -> Optional[float]:
        if energy is not None:
            # Find the closest time index in the DataFrame to the given time
            closest_time = self.dynamic_prices_df.index.get_loc(time)
            energy_price_kwh = self.dynamic_prices_df.iloc[closest_time]["price"]
            return energy_price_kwh * energy
        return None

    def calculate_price_vector(self, df: pd.DataFrame) -> pd.DataFrame:
        # Assuming the DataFrame has a datetime index and an 'energy' column
        prices = [
            self.calculate_price(time=row.name, energy=row["energy"])
            for index, row in df.iterrows()
        ]
        return pd.DataFrame(prices, index=df.index, columns=["price"])


class HOEPMarketTariff(Tariff):
    def __init__(
        self,
        hoep_file_path: str = "neuraflux/agency/datasets/hoep_interpolated_2023.csv",
    ):
        # Reading the dynamic prices from the provided CSV file
        self.dynamic_prices_df = pd.read_csv(
            hoep_file_path, index_col=0, parse_dates=True
        )
        self.dynamic_prices_df.index = self.dynamic_prices_df.index - dt.timedelta(
            hours=1
        )

    def calculate_price(
        self,
        time: Optional[dt.datetime] = None,
        power: Optional[float] = None,
        energy: Optional[float] = None,
        other_info: Dict[str, Any] = {},
    ) -> Optional[float]:
        if energy is not None:
            # Find the closest time index in the DataFrame to the given time
            closest_time = self.dynamic_prices_df.index.get_loc(time)
            energy_price_kwh = self.dynamic_prices_df.iloc[closest_time]["HOEP"]
            return energy_price_kwh * energy
        return None

    def calculate_price_vector(self, df: pd.DataFrame) -> pd.DataFrame:
        # Assuming the DataFrame has a datetime index and an 'energy' column
        prices = [
            self.calculate_price(time=row.name, energy=row[ENERGY_KEY])
            for index, row in df.iterrows()
        ]
        df["price"] = prices
        return df


@unique
class AvailableTariffsEnum(Enum):
    FLAT_RATE = FlatRateTariff
    HYDRO_QUEBEC_D = HydroQuebecDTariff
    ONTARIO_TOU = OntarioTOUTariff
    HYDRO_QUEBEC_M = HydroQuebecMTariff
    DYNNAMIC_PRICING = DynamicPricingTariff
    HOEP_MARKET = HOEPMarketTariff

    @classmethod
    def list_tariffs(cls):
        return list(map(lambda tariff: tariff.name.lower(), cls))

    @classmethod
    def get_tariff_class(cls, tariff_name: str):
        for tariff in cls:
            if tariff.name.lower() == tariff_name.lower():
                return tariff.value
        raise ValueError(f"No tariff found with name: {tariff_name}")


class TariffFactory:
    @staticmethod
    def create(tariff_name: str, **kwargs) -> Tariff:
        tariff_class = AvailableTariffsEnum.get_tariff_class(tariff_name)
        return tariff_class(**kwargs)
