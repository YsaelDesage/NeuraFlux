import datetime as dt
import pandas as pd

# General
IndexType = int | dt.datetime
SignalType = dict[str, (float | int | str)]
UidType = str

# Data Module
DBSignalType = dict[UidType, dict[IndexType, SignalType]]
DBScalingDictsType = dict[UidType, dict[str, tuple[float, float]]]
DBTrajectoriesType = dict[UidType, dict[IndexType, list[pd.DataFrame]]]
