import streamlit as st
import os
from neuraflux.agency.agent import Agent
from neuraflux.simulation import Simulation
from neuraflux.agency.data.data_module import DataModule
from neuraflux.agency.config.config_module import ConfigModule
from neuraflux.agency.control.control_module import ControlModule
from neuraflux.weather_ref import WeatherRef
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# from neuraflux.visualizations import plot_circle_subplot, plot_histograms
import pandas as pd
import numpy as np
import json


def retrieve_modules(sim_dir):
    data_module = DataModule.from_file(sim_dir)
    config_module = ConfigModule.from_file(sim_dir)
    control_module = ControlModule.from_file(sim_dir)
    return data_module, config_module, control_module


def retrieve_df(_data_module, cpm, weather_ref):
    return data_module.get_augmented_history(
        uid="1", controls_power_mapping=cpm, weather_ref=weather_ref
    )


def load_simulation_time_config_as_dict(sim_dir: str):
    filepath = os.path.join(sim_dir, "config.json")
    with open(filepath, "r") as f:
        simulation_config_dict = json.load(f)
    simulation_time_config_dict = simulation_config_dict["time"]
    return simulation_time_config_dict


def load_simulation_geo_config_as_dict(sim_dir: str):
    filepath = os.path.join(sim_dir, "config.json")
    with open(filepath, "r") as f:
        simulation_config_dict = json.load(f)
    simulation_geo_config_dict = simulation_config_dict["geography"]
    return simulation_geo_config_dict


sim_dir_1 = "sim_energy_storage"
sim_dir_2 = "sim_energy_storage_1"

sim_dict_1 = {}
sim_dict_2 = {}

for sim_dict, sim_dir in zip([sim_dict_1, sim_dict_2], [sim_dir_1, sim_dir_2]):
    sim_time_config = load_simulation_time_config_as_dict(sim_dir)
    sim_geo_config = load_simulation_geo_config_as_dict(sim_dir)

    lat = sim_geo_config["location_lat"]
    lon = sim_geo_config["location_lon"]
    alt = sim_geo_config["location_alt"]

    # Convert start string with format 2023-01-01_00-00-00 to datetime
    start_time = pd.to_datetime(
        sim_time_config["start_time"], format="%Y-%m-%d_%H-%M-%S"
    )
    end_time = pd.to_datetime(sim_time_config["end_time"], format="%Y-%m-%d_%H-%M-%S")

    data_module, config_module, control_module = retrieve_modules(sim_dir)
    weather_ref = WeatherRef.load_or_initialize(
        lat=lat,
        lon=lon,
        alt=alt,
        start=start_time,
        end=end_time,
    )
    agent_config = config_module.get_agent_config("1")
    cpm = agent_config.data.control_power_mapping

    sim = Simulation.from_directory(sim_dir)
    sim.load(sim_dir)

    df = retrieve_df(data_module, cpm, weather_ref)

    sim_dict["start_time"] = start_time
    sim_dict["end_time"] = end_time
    sim_dict["lat"] = lat
    sim_dict["lon"] = lon
    sim_dict["alt"] = alt
    sim_dict["data_module"] = data_module
    sim_dict["config_module"] = config_module
    sim_dict["control_module"] = control_module
    sim_dict["cpm"] = cpm
    sim_dict["agent_config"] = agent_config
    sim_dict["time_config"] = sim_time_config
    sim_dict["geo_config"] = sim_geo_config
    sim_dict["sim"] = sim
    sim_dict["weather_ref"] = weather_ref
    sim_dict["df"] = df

df_1 = sim_dict_1["df"]
df_2 = sim_dict_2["df"]

final_dfs = []
for sim_dict in [sim_dict_1, sim_dict_2]:
    df = sim_dict["df"]
    # Keep only data from january 3rd to 7
    df = df[
        (df.index >= dt.datetime(2023, 1, 3)) & (df.index <= dt.datetime(2023, 1, 8))
    ]

    data_module = sim_dict["data_module"]
    sub_df = df.copy()
    sim = sim_dict["sim"]
    data_module = sim_dict["data_module"]
    cpm = sim_dict["cpm"]
    weather_ref = sim_dict["weather_ref"]

    shadow_asset = sim.shadow_assets[0]
    shadow_df = shadow_asset.get_historical_data()
    shadow_df = shadow_df[shadow_df.index > dt.datetime(2023, 1, 2)]
    shadow_df = data_module.augment_df_with_all(
        uid="1",
        df=shadow_df,
        controls_power_mapping=cpm,
        weather_ref=weather_ref,
    )

    # Rename shadow df columns with "shadow_" prefix
    shadow_df = shadow_df.add_prefix("shadow_")
    augmented_df = pd.concat([sub_df, shadow_df], axis=1)
    money_cols = (
        ["tariff_$", "shadow_tariff_$"]
        if "tariff_$" in augmented_df.columns
        else ["price", "shadow_price"]
    )

    # Calculate rolling averages over 12 periods for cost columns
    augmented_df["agent_cost_col_rolling_avg"] = (
        augmented_df[money_cols[0]].rolling(window=12 * 24).mean()
    )
    augmented_df["shadow_cost_col_rolling_avg"] = (
        augmented_df[money_cols[1]].rolling(window=12 * 24).mean()
    )

    # Add cumulative profit column
    profit_col = "profit"
    # We use the same baseline for both
    if len(final_dfs) > 0:
        baseline_df = final_dfs[0]
    else:
        baseline_df = augmented_df
    augmented_df[profit_col] = augmented_df[money_cols[1]] - baseline_df[money_cols[0]]

    final_dfs.append(augmented_df)

final_df_1 = final_dfs[0]
final_df_2 = final_dfs[1]

# Keep only data from january 3rd to 7
final_df_1 = final_df_1[
    (final_df_1.index >= dt.datetime(2023, 1, 3))
    & (final_df_1.index <= dt.datetime(2023, 1, 8))
]
final_df_2 = final_df_2[
    (final_df_2.index >= dt.datetime(2023, 1, 3))
    & (final_df_2.index <= dt.datetime(2023, 1, 8))
]

# Add a column as the profit, linear between 0 and final profit
final_df_1["linear_profit"] = np.linspace(
    0, final_df_1[profit_col].sum(), len(final_df_1)
)
final_df_2["linear_profit"] = np.linspace(
    0, final_df_2[profit_col].sum(), len(final_df_2)
)

# Create a single plot with cumulative reward
fig = go.Figure(
    data=[
        go.Scatter(
            x=final_df_1.index,
            y=final_df_1[profit_col].cumsum(),
            mode="lines",
            name="Short-term (lower risk) horizon",
            # fill="tozeroy",
            line=dict(color="rgba(0, 0, 255, 0.2)"),
        ),
        go.Scatter(
            x=final_df_1.index,
            y=final_df_1["linear_profit"],
            mode="lines",
            #name="Average short horizon cumulative profit",
            showlegend=False,
            fill="tonexty",
            line=dict(color="blue", width=2, dash="dot"),
        ),
        go.Scatter(
            x=final_df_2.index,
            y=final_df_2[profit_col].cumsum(),
            mode="lines",
            name="Long-term (higher risk) horizon",
            # fill="tozeroy",
            line=dict(color="rgba(255, 0, 0, 0.2)"),
        ),
        go.Scatter(
            x=final_df_2.index,
            y=final_df_2["linear_profit"],
            mode="lines",
            #name="Average long horizon cumulative profit",
            showlegend=False,
            fill="tonexty",
            line=dict(color="red", width=2, dash="dot"),
        ),
    ]
)

# Add legend box to plot
fig.update_layout(
    legend=dict(
        title="Legend",  # You can give a title to the legend
        orientation="h",  # Horizontal legend
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)


# Update layout
fig.update_layout(template="plotly_white", showlegend=True)
fig.update_xaxes(title_text="Time", showgrid=False)
fig.update_yaxes(title_text="Agent Cumulative Profit ($)", showgrid=False)
# fig.update_yaxes(title_text="Cost ($)", row=1, col=1, showgrid=False)
# fig.update_yaxes(title_text="Agent Profit ($)", row=2, col=1, showgrid=False)
fig.show()
