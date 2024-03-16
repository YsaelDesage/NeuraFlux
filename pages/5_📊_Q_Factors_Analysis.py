import streamlit as st
import pandas as pd

# from neuraflux.visualizations import plot_circle_subplot, plot_histograms
import numpy as np
import plotly.express as px
from neuraflux.visualizations_q_factors import (
    plot_q_factors_as_circles,
    plot_q_factors_as_bars,
)
from neuraflux.agency.control.pareto import ParetoFrontier, plot_pareto_front_plotly

st.header("📊 Q-Factors Analysis")
if "df" not in st.session_state:
    st.warning("No simulation data loaded in memory.")
else:
    rl_config = st.session_state.agent_config.control.reinforcement_learning
    control_module = st.session_state.control_module
    df = st.session_state.df
    scaled_df = st.session_state.data_module.scale_data("1", df)
    rl_config = st.session_state.agent_config.control.reinforcement_learning
    control_module = st.session_state.control_module
    df = control_module.augment_df_with_q_factors(
        uid="1", data=df, scaled_data=scaled_df, rl_config=rl_config
    )

    # TIME PERIOD SELECTION
    col01, col02, col03 = st.columns((3, 10, 1))
    controller = col01.selectbox(
        "Select controller",
        [
            str(i)
            for i in range(
                1,
                st.session_state.agent_config.control.reinforcement_learning.n_controllers
                + 1,
            )
        ],
    )
    if "internal_energy" in df.columns:
        n_rewards = 1
    elif "temperature_1" in df.columns:
        n_rewards = 3
    else:
        n_rewards = 1
    start_idx, end_idx = col02.select_slider(
        "Select date range",
        df.index[6:],
        value=(df.index[6], df.index[-1]),
        format_func=lambda x: x.strftime("%Y/%m/%d %H:%M"),
    )
    if st.button("Show analysis"):
        tabs = st.tabs(["Reward " + str(i) for i in range(1, n_rewards + 1)])
        sub_df = df.copy()[(df.index >= start_idx) & (df.index <= end_idx)]

        for n_reward in range(1, n_rewards + 1):
            with tabs[n_reward - 1]:
                st.write("### Q-Factors Analysis")

                st.write("##### Current Relative Values")

                # Relative circles plot at time t
                Q1, Q2, Q3 = sub_df.loc[start_idx][
                    [
                        f"Q{n_reward}_C{controller}_U1",
                        f"Q{n_reward}_C{controller}_U2",
                        f"Q{n_reward}_C{controller}_U3",
                    ]
                ]
                Q_factors = [Q1, Q2, Q3]
                if f"Q{n_reward}_C{controller}_U4" in sub_df.columns:
                    Q4 = sub_df.loc[start_idx][f"Q{n_reward}_C{controller}_U4"]
                    Q_factors.append(Q4)
                    Q5 = sub_df.loc[start_idx][f"Q{n_reward}_C{controller}_U5"]
                    Q_factors.append(Q5)

                if len(Q_factors) == 3:
                    titles = ["0: Sell", "1: Do Nothing", "2: Buy"]
                elif len(Q_factors) == 5:
                    titles = ["Cool2", "Cool1", "Off", "Heat1", "Heat2"]
                plot = plot_q_factors_as_circles(Q_factors, titles=titles)
                st.plotly_chart(plot, use_container_width=True)

                st.write("##### Values Evolution")
                # Loop over indexes after idx and plot Q-factors
                max_plots = min(12, len(sub_df))
                plots = []
                global_min = min(
                    sub_df.iloc[0:max_plots, :][f"Q{n_reward}_C{controller}_U1"].min(),
                    sub_df.iloc[0:max_plots, :][f"Q{n_reward}_C{controller}_U2"].min(),
                    sub_df.iloc[0:max_plots, :][f"Q{n_reward}_C{controller}_U3"].min(),
                )
                global_max = max(
                    sub_df.iloc[0:max_plots, :][f"Q{n_reward}_C{controller}_U1"].max(),
                    sub_df.iloc[0:max_plots, :][f"Q{n_reward}_C{controller}_U2"].max(),
                    sub_df.iloc[0:max_plots, :][f"Q{n_reward}_C{controller}_U3"].max(),
                )
                for i in range(0, max_plots):
                    Q1, Q2, Q3 = sub_df.loc[start_idx][
                        [
                            f"Q{n_reward}_C{controller}_U1",
                            f"Q{n_reward}_C{controller}_U2",
                            f"Q{n_reward}_C{controller}_U3",
                        ]
                    ]
                    Q_factors = [Q1, Q2, Q3]
                    if f"Q{n_reward}_C{controller}_U4" in sub_df.columns:
                        Q4 = sub_df.loc[sub_df.index[i]][
                            f"Q{n_reward}_C{controller}_U4"
                        ]
                        Q_factors.append(Q4)
                        Q5 = sub_df.loc[sub_df.index[i]][
                            f"Q{n_reward}_C{controller}_U5"
                        ]
                        Q_factors.append(Q5)

                    plots.append(
                        plot_q_factors_as_bars(
                            Q_factors,
                            sub_df.index[i],
                            fixed_min=global_min,
                            fixed_max=global_max,
                        )
                    )

                col11, col12, col13, col14 = st.columns(4)

                for i, plot in enumerate(plots):
                    # Swith between columns (4 plots per row)
                    if i % 4 == 0:
                        col11.plotly_chart(plot, use_container_width=True)
                    elif i % 4 == 1:
                        col12.plotly_chart(plot, use_container_width=True)
                    elif i % 4 == 2:
                        col13.plotly_chart(plot, use_container_width=True)
                    elif i % 4 == 3:
                        col14.plotly_chart(plot, use_container_width=True)

        rl_training_logs = control_module.get_rl_training_logs("1")
        str_format = "%m/%d %H:00"
        st.write("### TD Error Training Evolution")
        tabs = st.tabs([k.strftime(str_format) for k in rl_training_logs.keys()])
        for i, k in enumerate(rl_training_logs.keys()):
            with tabs[i]:
                real_errors = rl_training_logs[k]["real_td_errors"]
                sim_errors = rl_training_logs[k]["sim_td_errors"]

                # concatenate list of lists
                real_errors = np.concatenate(real_errors).ravel()
                sim_errors = np.concatenate(sim_errors).ravel()
                # Create a DataFrame from the arrays
                df = pd.DataFrame(
                    {
                        "Iteration": np.arange(len(real_errors)),
                        "Real Errors": real_errors,
                        "Sim Errors": sim_errors,
                    }
                )

                # Plot using Plotly Express
                fig = px.line(
                    df,
                    x="Iteration",
                    y=["Sim Errors", "Real Errors"],
                )
                st.plotly_chart(fig)

        st.write("### Multiobjective Pareto Front")
        if "temperature_1" in sub_df.columns:
            fig = plot_pareto_front_plotly(sub_df, start_idx)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Single objective Product - skipping analysis.")

        shadow_asset = st.session_state.sim.shadow_assets[0]
        shadow_df = shadow_asset.get_historical_data()
        st.write(shadow_df)

        st.write(sub_df)
