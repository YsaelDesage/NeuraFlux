import pandas as pd


def update_scaling_dict_from_df(
    df: pd.DataFrame,
    scaling_dict: dict[str, tuple[float, float]],
    columns_to_scale: list[str] = [],
) -> dict[str, tuple[float, float]]:
    """Updates a scaling dictionary based on the min and max values of a
    DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        scaling_dict (dict[str, tuple[float, float]]): Dictionary where keys
        are column names and values are a tuple of (min_value, max_value) for
        scaling.

    Returns:
        dict[str, tuple[float, float]]: Updated scaling dictionary.
    """
    df = df.copy()

    for col in columns_to_scale:
        col_min = df[col].min()
        col_max = df[col].max()

        # If column already in dictionary, update the min and max values
        if col in scaling_dict:
            existing_min, existing_max = scaling_dict[col]
            scaling_dict[col] = (
                min(existing_min, col_min),
                max(existing_max, col_max),
            )
        else:
            scaling_dict[col] = (col_min, col_max)

    # TODO: Handle fields to never scale
    # scaling_dict = {"internal_energy": scaling_dict["internal_energy"]}

    return scaling_dict


def scale_df_based_on_scaling_dict(
    df: pd.DataFrame, scaling_dict: dict, scale_to_minus1_and_1: bool = False
) -> pd.DataFrame:
    """
    Scales a DataFrame based on min/max values provided in a dictionary.
    Unspecified columns are left unchanged.

    Args:
    - df (pd.DataFrame): The input DataFrame.
    - scaling_dict (dict): Dictionary where keys are column names and values
    are a tuple of (min_value, max_value) for scaling.
    - scale_to_minus1_and_1 (bool): Whether to scale between -1 and 1 instead
    of 0 and 1.

    Returns:
    - pd.DataFrame: Scaled DataFrame.
    """
    scaled_df = df.copy()

    for col in df.columns:
        if col in scaling_dict:
            min_val, max_val = scaling_dict[col]

            # Check for the case where min and max are the same
            if min_val == max_val:
                scaled_df[col] = 0  # or any other default value
            else:
                # Apply min-max scaling formula
                x_scaled = (df[col] - min_val) / (max_val - min_val)

                # If scaling between -1 and 1 is desired
                if scale_to_minus1_and_1:
                    x_scaled = 2 * x_scaled - 1

                scaled_df[col] = x_scaled

    return scaled_df
