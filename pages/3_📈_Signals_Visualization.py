import streamlit as st
import os
from neuraflux.agency.agent import Agent
from neuraflux.agency.data.data_module import DataModule
from neuraflux.agency.config.config_module import ConfigModule
#from neuraflux.visualizations import plot_circle_subplot, plot_histograms
import pandas as pd
import numpy as np
import json
import plotly.express as px


st.header("📈 Signals Visualization")
if "df" not in st.session_state:
    st.warning("No simulation data loaded in memory.")
else:
    df = st.session_state.df

    signal_col1, _, signal_col2 = st.columns((5, 1, 10))
    available_signals = df.columns
    plot_col = signal_col1.selectbox("Select column", available_signals)
    start, end = signal_col2.select_slider(
        "Select date range",
        df.index,
        value=(df.index[0], df.index[-1]),
        format_func=lambda x: x.strftime("%Y/%m/%d %H:%M"),
    )
    if st.button("Show data"):
        filtered_df = df.copy()[(df.index >= start) & (df.index <= end)]
        
        # Plot desired column with a line chart with a white bg in plotly express
        fig = px.line(filtered_df, x=filtered_df.index, y=plot_col)
        st.plotly_chart(fig, use_container_width=True)



        st.write(filtered_df)



#     # -----------------------------------------------------------------------
#     # - TRAJECTORIES
#     # -----------------------------------------------------------------------
#     st.write("#### Simulated Trajectories")
#     idx = st.select_slider(
#         "Select timestamp",
#         history.index,
#         value=history.index[0],
#         format_func=lambda x: x.strftime("%Y/%m/%d %H:%M"),
#     )
#     trajectory_col1, trajectory_col2 = st.columns((1, 2))
#     trajectories = data_module.get_trajectory_data(str(uid), idx)
#     traj_idx = trajectory_col1.selectbox(
#         "Select trajectory", range(1, len(trajectories) + 1)
#     )
#     tau = agent.get_dataframe_with_Q_factors(trajectories[traj_idx - 1])
#     st.write(tau)
#     df = data_module.augment_dataframe_with_return(
#         df=tau,
#         reward_col="reward_DR",
#         Q_cols=["Q_0", "Q_1", "Q_2"],
#         return_col="return",
#     )
#     st.plotly_chart(plot_histograms(df, ["return"]))

# training_real_errors = data_module.get_real_rl_training_data(str(uid))
# training_virtual_errors = data_module.get_virtual_rl_training_data(str(uid))

# training_real_errors_continuous = []
# for i in range(len(training_real_errors["test_timestamp"])):
#     training_real_errors_continuous = training_real_errors_continuous + training_real_errors["test_timestamp"][i]

# st.line_chart(training_real_errors_continuous)
# #st.write(training_virtual_errors)

# st.write("#### Q-values and analysis")
# fig = plot_circle_subplot(0.8, -0.2, 0.5)
# st.plotly_chart(fig, use_container_width=True)