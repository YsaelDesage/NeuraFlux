import streamlit as st
import datetime as dt
import pandas as pd
from neuraflux.visualizations_q_factors import create_financial_performance_plot
from neuraflux.utils_visualizations import plot_temperature_power
from neuraflux.assets.electric_vehicle import ElectricVehicle

st.header("💰 Reward Analysis")
if "df" not in st.session_state:
    st.warning("No simulation data loaded in memory.")
else:
    rl_config = st.session_state.agent_config.control.reinforcement_learning
    control_module = st.session_state.control_module
    data_module = st.session_state.data_module

    df = st.session_state.df
    # df = df[df.index >= dt.datetime(2023, 1, 2)]
    scaled_df = st.session_state.data_module.scale_data("1", df)
    rl_config = st.session_state.agent_config.control.reinforcement_learning
    control_module = st.session_state.control_module
    df = control_module.augment_df_with_q_factors(
        uid="1", data=df, scaled_data=scaled_df, rl_config=rl_config
    )

    # TIME PERIOD SELECTION
    col01, col02 = st.columns((10, 4))
    start_idx, end_idx = col01.select_slider(
        "Select date range",
        df.index,
        value=(df.index[0], df.index[-1]),
        format_func=lambda x: x.strftime("%Y/%m/%d %H:%M"),
    )
    if st.button("Show analysis"):
        sub_df = df.copy()[(df.index >= start_idx) & (df.index <= end_idx)]
        shadow_asset = st.session_state.sim.shadow_assets[0]
        shadow_df = shadow_asset.get_historical_data()
        shadow_df = shadow_df[shadow_df.index > dt.datetime(2023, 1, 2)]
        shadow_df = data_module.augment_df_with_all(
            uid="1",
            df=shadow_df,
            controls_power_mapping=st.session_state.cpm,
            weather_ref=st.session_state.weather_ref,
        )

        # Rename shadow df columns with "shadow_" prefix
        shadow_df = shadow_df.add_prefix("shadow_")
        augmented_df = pd.concat([sub_df, shadow_df], axis=1)
        is_reward = False
        if "tariff_$" in augmented_df.columns and not isinstance(
            shadow_asset, ElectricVehicle
        ):
            money_cols = ["tariff_$", "shadow_tariff_$"]
        elif not isinstance(shadow_asset, ElectricVehicle):
            money_cols = ["price", "shadow_price"]
        else:
            money_cols = ["reward_COST", "shadow_reward_COST"]
            is_reward = True
        fig = create_financial_performance_plot(
            augmented_df, money_cols[0], money_cols[1], color="gold", reward=is_reward
        )

        _, plot_col, _ = st.columns((1, 3, 1))
        plot_col.plotly_chart(fig, use_container_width=True)

        st.write(shadow_df)
        st.write(sub_df)

        st.write(shadow_df["shadow_reward_GHG"].sum())
        st.write(sub_df["reward_GHG"].sum())

        if "temperature" in sub_df:
            df_T = augmented_df[
                (augmented_df.index >= start_idx)
                & (augmented_df.index <= start_idx + dt.timedelta(hours=15))
            ]
            df_T["temperature"] = (
                df_T["temperature_1"] + df_T["temperature_2"] + df_T["temperature_3"]
            ) / 3
            df_T["baseline_temperature"] = (
                df_T["shadow_temperature_1"]
                + df_T["shadow_temperature_2"]
                + df_T["shadow_temperature_3"]
            ) / 3

            st.subheader("Temperature and Power")
            fig = plot_temperature_power(
                df_T,
                "temperature",
                "baseline_temperature",
                "heat_setpoint",
                "cool_setpoint",
                "power",
                "shadow_power",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Scenario Tree")
