import streamlit as st
import os
from neuraflux.agency.agent import Agent
from neuraflux.simulation import Simulation
from neuraflux.agency.data.data_module import DataModule
from neuraflux.agency.config.config_module import ConfigModule
from neuraflux.agency.control.control_module import ControlModule
from neuraflux.weather_ref import WeatherRef
import datetime as dt

# from neuraflux.visualizations import plot_circle_subplot, plot_histograms
import pandas as pd
import numpy as np
import json


@st.cache_data
def retrieve_modules(sim_dir):
    data_module = DataModule.from_file(sim_dir)
    config_module = ConfigModule.from_file(sim_dir)
    control_module = ControlModule.from_file(sim_dir)
    return data_module, config_module, control_module


# @st.cache_data
def retrieve_df(_data_module, cpm, weather_ref):
    return data_module.get_augmented_history(
        uid="1", controls_power_mapping=cpm, weather_ref=weather_ref
    )


@st.cache_data
def load_simulation_time_config_as_dict(sim_dir: str):
    filepath = os.path.join(sim_dir, "config.json")
    with open(filepath, "r") as f:
        simulation_config_dict = json.load(f)
    simulation_time_config_dict = simulation_config_dict["time"]
    return simulation_time_config_dict


@st.cache_data
def load_simulation_geo_config_as_dict(sim_dir: str):
    filepath = os.path.join(sim_dir, "config.json")
    with open(filepath, "r") as f:
        simulation_config_dict = json.load(f)
    simulation_geo_config_dict = simulation_config_dict["geography"]
    return simulation_geo_config_dict


st.title("⚙️ Configuration")

st.write("### 📁 Simulation Directory")
col00, col01 = st.columns(2)

# Get list of possible directories from all folders containing "sim" in the name
sim_dir_list = [d for d in os.listdir() if "sim" in d]

# Keep sim directory in memory, use None by default if never executed
if "sim_dir" not in st.session_state:
    st.session_state.sim_dir = None

# Convert sim_dir to index to initialize proper value in selectbox
if st.session_state.sim_dir is not None:
    sim_dir_index = sim_dir_list.index(st.session_state.sim_dir)
else:
    sim_dir_index = None
sim_dir = col00.selectbox(
    "Select simulation directory", sim_dir_list, index=sim_dir_index
)
st.session_state.sim_dir = sim_dir

if st.session_state.sim_dir is not None:
    if "data_module" not in st.session_state:
        with st.spinner("Loading modules ..."):
            sim_time_config = load_simulation_time_config_as_dict(sim_dir)
            sim_geo_config = load_simulation_geo_config_as_dict(sim_dir)

            lat = sim_geo_config["location_lat"]
            lon = sim_geo_config["location_lon"]
            alt = sim_geo_config["location_alt"]

            # Convert start string with format 2023-01-01_00-00-00 to datetime
            start_time = pd.to_datetime(
                sim_time_config["start_time"], format="%Y-%m-%d_%H-%M-%S"
            )
            end_time = pd.to_datetime(
                sim_time_config["end_time"], format="%Y-%m-%d_%H-%M-%S"
            )

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

            st.session_state.start_time = start_time
            st.session_state.end_time = end_time
            st.session_state.lat = lat
            st.session_state.lon = lon
            st.session_state.alt = alt
            st.session_state.data_module = data_module
            st.session_state.config_module = config_module
            st.session_state.control_module = control_module
            st.session_state.cpm = cpm
            st.session_state.agent_config = agent_config
            st.session_state.time_config = sim_time_config
            st.session_state.geo_config = sim_geo_config
            st.session_state.sim = sim
            st.session_state.weather_ref = weather_ref

    if "df" not in st.session_state:
        with st.spinner("Loading data ..."):
            df = retrieve_df(data_module, cpm, weather_ref)
            st.session_state.df = df

    if "df" in st.session_state:
        col11, col12 = st.columns((2, 2))
        col11.write("#### 🕓 Time")
        col12.write("#### 🌍 Geography")

        col21, col23 = st.columns((2, 2))

        col21.date_input("Simulation start", st.session_state.start_time, disabled=True)
        col21.date_input("Simulation end", st.session_state.end_time, disabled=True)
        col21.number_input(
            "Simulation steps (s)",
            st.session_state.time_config["step_size_s"],
            disabled=True,
        )

        with col23:
            st.map(
                pd.DataFrame(
                    {"lat": [st.session_state.lat], "lon": [st.session_state.lon]}
                )
            )
            st.write(
                " 📍 ",
                "**Lat**: ",
                st.session_state.lat,
                "  **Lon**: ",
                st.session_state.lon,
                "  **Alt**: ",
                st.session_state.alt,
                "m",
            )

        st.write("### 🕵🏼 Agent Configuration")
        with st.expander("Show raw configuration"):
            st.write(st.session_state.agent_config.model_dump())
