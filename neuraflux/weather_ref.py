import datetime as dt
import json
import os
from meteostat import Hourly, Point  # type: ignore


from neuraflux.global_variables import DT_STR_FORMAT


class WeatherRef:
    """Weather data reference for the simulation, provides weather data
    based on true historical values at a target location."""

    def __init__(
        self,
        start: dt.datetime,
        end: dt.datetime,
        lat: float,
        lon: float,
        alt: float = 0,
    ):
        """Initialise the weather reference, and prepare data based
        on the specified parameters.

        Args:
            start (dt.datetime): Start of the simulation.
            end (dt.datetime): End of the simulation.
            lat (float): Latitude of the target location.
            lon (float): Longitude of the target location.
            alt (float, optional): Altitude of the target location.
                                   Defaults to 0.
        """

        # Store attributes
        self.lat = lat
        self.lon = lon
        self.alt = alt

        # Fetch historical weather data for specified location
        try:
            location = Point(lat=lat, lon=lon, alt=alt)
            retriever = Hourly(location, start, end)
            historical_hourly_data = retriever.fetch()

            # Interpolate data to 5 minutes interval
            self.data = historical_hourly_data.asfreq("5T").interpolate()[["temp"]]
        except:
            print("WARNING ! Unable to load historical data.")
            self.data = None

    def get_temperature_at_time(self, time: dt.datetime) -> float:
        """Returns the temperature at the specified time.

        Args:
            time (dt.datetime): The time to get the temperature at.

        Returns:
            float: The temperature at the specified time.
        """
        # Return def testing value if no data available
        if self.data is None:
            return -10
        iloc_idx = self.data.index.get_indexer([time], method="nearest")
        temperature = float(self.data["temp"].iloc[iloc_idx].values[0])
        return temperature

    def to_file(self, directory: str) -> None:
        file_path = os.path.join(directory, "weather_ref.json")
        attr_dict = {
            "start": self.data.index[0].strftime(DT_STR_FORMAT),
            "end": self.data.index[-1].strftime(DT_STR_FORMAT),
            "lat": self.lat,
            "lon": self.lon,
            "alt": self.alt,
        }
        with open(file_path, "w") as f:
            json.dump(attr_dict, f, indent=4)

    @classmethod
    def from_file(cls, directory: str) -> "WeatherRef":
        file_path = os.path.join(directory, "weather_ref.json")
        with open(file_path, "r") as f:
            attr_dict = json.load(f)
        start = dt.datetime.strptime(attr_dict["start"], DT_STR_FORMAT)
        end = dt.datetime.strptime(attr_dict["end"], DT_STR_FORMAT)
        lat = attr_dict["lat"]
        lon = attr_dict["lon"]
        alt = attr_dict["alt"]
        self = cls(start=start, end=end, lat=lat, lon=lon, alt=alt)
        return self

    @classmethod
    def load_or_initialize(
        cls,
        start: dt.datetime,
        end: dt.datetime,
        lat: float,
        lon: float,
        alt: float = 0,
        directory: str = "",
    ) -> "WeatherRef":
        file_path = os.path.join(directory, "weather_ref.json")
        if os.path.isfile(file_path):
            # Load from file if it exists
            return cls.from_file(directory)

        # Else, initialize a new instance with the provided parameters
        return cls(
            start=start,
            end=end,
            lat=lat,
            lon=lon,
            alt=alt,
        )
