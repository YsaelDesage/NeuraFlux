from neuraflux.agency.module import Module
from neuraflux.local_typing import UidType
from neuraflux.schemas.config import ModelEvaluationMetrics
import datetime as dt
import pandas as pd
from abc import ABCMeta, abstractmethod
from neuraflux.global_variables import CONTROL_KEY


class PredictionModel(metaclass=ABCMeta):
    def __init__(
        self, signal_inputs: dict[str, int], signal_outputs: list[str]
    ) -> None:
        self.signal_inputs = signal_inputs
        self.signal_outputs = signal_outputs

        # Add controls to signal inputs, if not already present
        if CONTROL_KEY not in self.signal_inputs:
            self.signal_inputs[CONTROL_KEY] = 0

    @abstractmethod
    def train(self, historical_signals: pd.DataFrame) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self, signals_input: pd.DataFrame, new_index: int | dt.datetime
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self, historical_signals: pd.DataFrame
    ) -> ModelEvaluationMetrics:
        raise NotImplementedError


class PredictionModule(Module):
    def assign_model(self, uid: int | str, model: PredictionModel) -> None:
        """Assigns a model to the specified asset uid.

        Args:
            uid (Union[int, str]): The unique identifier of the asset.
            model (PredictionModel): The model object to assign to this
                                     instance.
        """
        self.models[uid] = model

    def train_model(
        self, uid: int | str, historical_signals: pd.DataFrame
    ) -> None:
        """Train the model for the specified asset.

        Args:
            uid (Union[int, str]): The unique identifier of the asset.
            historical_signals (pd.DataFrame): The historical signals
                                               to train the model on.
        """
        self.models[uid].train(historical_signals)

    def get_model_prediction(
        self,
        uid: int | str,
        signals_df: pd.DataFrame,
        current_index: int | dt.datetime,
        new_index: int | dt.datetime,
    ) -> pd.DataFrame:
        """Get the model's prediction for the specified asset.

        Args:
            uid (Union[int, str]): The unique identifier of the asset.
            signals_df (pd.DataFrame): The signals to predict from.
            new_index (Union[int, dt.datetime]): The index of the prediction.

        Returns:
            pd.DataFrame: The model's prediction.
        """

        prediction_df: pd.DataFrame = self.models[uid].predict(
            signals_df.copy(), current_index, new_index
        )
        return prediction_df

    def models_retraining_loop(self) -> None:
        pass

    # ----------------------------------------------------------------------
    # PRIVATE METHODS
    # ----------------------------------------------------------------------
    def _initialize_data_structures(self) -> None:
        self.models: dict[UidType, PredictionModel] = {}
        self.models_performance: dict[
            UidType, list[ModelEvaluationMetrics]
        ] = {}
