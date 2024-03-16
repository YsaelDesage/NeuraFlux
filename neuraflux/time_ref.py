import datetime as dt
import json
import os

from neuraflux.global_variables import DT_STR_FORMAT


class TimeRef:
    def __init__(
        self,
        start_time_utc: dt.datetime,
    ) -> None:
        """Initialise the time module.

        Args:
            time_utc (dt.datetime): The initial time, in UTC.
            directory (str, optional): The directory to save the time
        """
        self.time_utc = start_time_utc
        self.initial_time_utc = start_time_utc

    def get_time_utc(self) -> dt.datetime:
        """Get the current time, in UTC.

        Returns:
            dt.datetime: Current time, in UTC
        """
        return self.time_utc

    def get_time_utc_as_str(self) -> str:
        """Get the current time, in UTC, as a string.

        Returns:
            str: Current time, in UTC, as a string
        """
        return self.time_utc.strftime(DT_STR_FORMAT)

    def get_initial_time_utc(self) -> dt.datetime:
        """Get the starting time, in UTC.

        Returns:
            dt.datetime: Starting time, in UTC
        """
        return self.initial_time_utc

    def get_initial_time_utc_as_str(self) -> str:
        """Get the starting time, in UTC, as a string.

        Returns:
            str: Starting time, in UTC, as a string
        """
        return self.initial_time_utc.strftime(DT_STR_FORMAT)

    def increment_time(self, delta: dt.timedelta | int) -> None:
        """Increment the current time by the specified delta.

        Args:
            delta (Union[dt.timedelta, int]): The delta to increment
            the time by. Integers will be converted to seconds.
        """
        if isinstance(delta, int):
            delta = dt.timedelta(seconds=delta)
        self.time_utc += delta

    def to_file(self, directory: str) -> None:
        file_path = os.path.join(directory, "time_ref.json")
        time_dict = {
            "time_utc": self.time_utc.strftime(DT_STR_FORMAT),
            "initial_time_utc": self.initial_time_utc.strftime(DT_STR_FORMAT),
        }
        with open(file_path, "w") as f:
            json.dump(time_dict, f, indent=4)

    @classmethod
    def from_file(cls, directory: str) -> "TimeRef":
        file_path = os.path.join(directory, "time_ref.json")
        with open(file_path, "r") as f:
            time_dict = json.load(f)
        time_utc = dt.datetime.strptime(time_dict["time_utc"], DT_STR_FORMAT)
        initial_time_utc = dt.datetime.strptime(
            time_dict["initial_time_utc"], DT_STR_FORMAT
        )
        self = cls(time_utc=time_utc)
        self.initial_time_utc = initial_time_utc
        return self

    @classmethod
    def load_or_initialize(
        cls, start_time_utc: dt.datetime, directory: str = ""
    ) -> "TimeRef":
        file_path = os.path.join(directory, "time_ref.json")
        # Check if the file exists
        if os.path.isfile(file_path):
            # Load from the file if it exists
            return cls.from_file(directory)
        # Else, initialize a new instance with the current or provided time
        return cls(start_time_utc=start_time_utc)
