import dill
import os


class Module:
    """
    A base class for all modules in the NeuraFlux framework.

    Methods:
        to_file: Saves the module to a file.
        from_file: Loads the module from a file.
    """

    def __init__(self):
        self._initialize_data_structures()

    def to_file(self, directory: str = "") -> None:
        """
        Saves the module to a file.

        Args:
            directory (str): The directory to save the file in.

        Returns:
            None
        """
        filepath = os.path.join(directory, self.__class__.__name__)
        with open(filepath, "wb") as f:
            dill.dump(self, f)

    @classmethod
    def from_file(cls, directory: str = "") -> "Module":
        filepath = os.path.join(directory, cls.__name__)
        try:
            with open(filepath, "rb") as f:
                instance = dill.load(f)
        except FileNotFoundError:
            instance = None
        return instance

    @classmethod
    def load_or_initialize(cls, directory: str = ""):
        instance = cls.from_file(directory)
        if instance is None:
            instance = cls()
        return instance

    def _initialize_data_structures(self):
        pass
