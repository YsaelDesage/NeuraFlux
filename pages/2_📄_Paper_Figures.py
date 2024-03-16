import streamlit as st
import datetime as dt

# from neuraflux.visualizations import plot_circle_subplot, plot_histograms
import pandas as pd
import numpy as np
import json
import plotly.graph_objs as go
from neuraflux.agency.control.scenario_tree import ScenarioTree, Node


def compute_scenario_tree_metrics(df):
    df = df.copy()
    df["cum_reward"] = df["reward_COST"].cumsum()
    df = df.iloc[6:]
    df["Q_max"] = df[["Q1_C1_U1", "Q1_C1_U2", "Q1_C1_U3"]].max(axis=1)
    # Set arg max to 1 if Q1_C1_U1 is the max, 2 if Q1_C1_U2 is the max, etc.
    df["argmax"] = (
        df[["Q1_C1_U1", "Q1_C1_U2", "Q1_C1_U3"]].idxmax(axis=1).str[-1].astype(int) - 1
    )
    df["returns"] = df["cum_reward"] + df["Q_max"]
    return df


st.header("📄 Paper Figures")
logs = st.session_state.control_module.get_rl_training_logs("1")
key = st.selectbox("Select trainings", list(logs.keys()))

real_err = logs[key]["real_td_errors"]
sim_err = logs[key]["sim_td_errors"]

# Combine all individual lists of errors into a single big one
real_err = [item for sublist in real_err for item in sublist]
sim_err = [item for sublist in sim_err for item in sublist]

err_df = pd.DataFrame({"real error": real_err, "sim error": sim_err})

# Plot both errors over time in plotly, with the real error on top (not using express)
# Make simulation error more transparent
fig = go.Figure(
    data=[
        go.Scatter(
            x=err_df.index,
            y=err_df["sim error"],
            name="Sim Error",
            mode="lines",
            line=dict(color="green"),
            opacity=0.6,
        ),
        go.Scatter(
            x=err_df.index,
            y=err_df["real error"],
            name="Real Error",
            mode="lines",
            line=dict(color="blue"),
            opacity=0.6,
        ),
    ]
)
# Add X and y labels for iters and error
fig.update_xaxes(title_text="Iteration")
fig.update_yaxes(title_text="Error")
st.plotly_chart(fig, use_container_width=True)


df = st.session_state.df
# df = df[df.index >= dt.datetime(2023, 1, 2)]
scaled_df = st.session_state.data_module.scale_data("1", df)
rl_config = st.session_state.agent_config.control.reinforcement_learning
control_module = st.session_state.control_module
df = control_module.augment_df_with_q_factors(
    uid="1", data=df, scaled_data=scaled_df, rl_config=rl_config
)

trajectories = st.session_state.data_module.get_trajectory_data(
    "1", dt.datetime(2023, 1, 3, 18, 15)
)
metrics = []
for trajectory in trajectories:
    scaled_trajectory = st.session_state.data_module.scale_data("1", trajectory)
    q_trajectory = control_module.augment_df_with_q_factors(
        uid="1", data=trajectory, scaled_data=scaled_trajectory, rl_config=rl_config
    )

    metrics.append(compute_scenario_tree_metrics(q_trajectory))

st.subheader("Scenario Tree")

metrics = metrics[-3:]
for m in metrics:
    st.write(m)
m1 = metrics[0]
m2 = metrics[1]
m3 = metrics[2]
ghg1 = round((m1["energy"] * m1["emission_rate"]).sum(), 2)
ghg2 = round((m2["energy"] * m2["emission_rate"]).sum(), 2)
ghg3 = round((m3["energy"] * m3["emission_rate"]).sum(), 2)
tree = ScenarioTree(
    "initial state",
    properties={
        "name": "INITIAL STATE",
        "Time": "2023-01-03 18:15",
        "State of Charge (%)": 90,
    },
)
state1_node = Node(
    "state1",
    0,
    properties={
        "name": "OPTION 1 (Full Discharge)",
        "Max Profit ($)": round(m1["returns"].max(), 2),
        "Emissions (gCO2)": ghg1,
    },
)
state2_node = Node(
    "state2",
    0,
    properties={
        "name": "OPTION 2 (Idle)",
        "Max Profit ($)": round(m2["returns"].max(), 2),
        "Emissions (gCO2)": ghg2,
    },
)
state3_node = Node(
    "state3",
    0,
    properties={
        "name": "OPTION 3 (Full Charge)",
        "Max Profit ($)": round(m3["returns"].max(), 2),
        "Emissions (gCO2)": ghg3,
    },
)

tree.insert("initial state", "U1", [state1_node], [10], [1])
tree.insert("initial state", "U2", [state2_node], [200], [1])
tree.insert("initial state", "U3", [state3_node], [40], [1])

tree_viz = tree.to_graphviz()
st.graphviz_chart(tree_viz)

cols = st.columns((7, 7, 7, 3))

for i in range(3):
    m = metrics[i]
    col = cols[i]

    data = m["returns"].values

    # Create a histogram with a golden color
    fig = go.Figure(
        data=[
            go.Histogram(
                x=data,
                histnorm="probability",
                nbinsx=6,
                marker_color="rgba(255, 215, 0, 0.6)",  # golden color, semi-transparent
            )
        ]
    )

    # Update layout for a white background and reduced top margin
    fig.update_layout(
        plot_bgcolor="white",  # white background for the plot
        paper_bgcolor="white",  # white background for the paper
        margin=dict(
            t=0, l=10, b=10, r=10
        ),  # top, left, bottom, right margins in pixels
        width=800,  # Width of the figure in pixels
        height=140,  # Height of the figure in pixels
        xaxis=dict(showgrid=False, title_text="Expected Profit ($)"),
        yaxis=dict(showgrid=False, title_text="Probability"),
    )

    # If you're using Streamlit, use this line to display the plot:
    col.plotly_chart(fig, use_container_width=True)


tree = ScenarioTree(
    "initial state",
    properties={
        "name": "INITIAL STATE",
        "Time": "2023-01-01 18:15",
        "EV Battery (kWh)": 62.5,
    },
)

E0 = 62.5
dE = 2.08
e1 = round(E0 - dE, 2)
e2 = round(E0, 2)
e3 = round(E0 + dE, 2)
g1 = round(1.84 * -dE, 2)
g2 = round(0, 2)
g3 = round(1.84 * dE, 2)
r1 = round(dE, 2)
r2 = round(0, 2)
r3 = round(-dE, 2)

state1_node = Node(
    "state1",
    0,
    properties={
        "EV Battery (kWh)": e1,
        "Reward ($)": r1,
        "Emissions (gCO2)": g1,
    },
)
state2_node = Node(
    "state2",
    0,
    properties={
        "EV Battery (kWh)": e2,
        "Reward ($)": r2,
        "Emissions (gCO2)": g2,
    },
)
state3_node = Node(
    "state3",
    0,
    properties={
        "EV Battery (kWh)": e3,
        "Reward ($)": r3,
        "Emissions (gCO2)": g3,
        "+ Terminal Q-Factors": "",
    },
)

e4 = round(e1 - dE, 2)
e5 = round(e1, 2)
g4 = round(1.84 * -dE, 2)
g5 = round(0, 2)
r4 = round(dE, 2)
r5 = round(0, 2)

state4_node = Node(
    "state4",
    0,
    properties={
        "EV Battery (kWh)": e4,
        "Reward ($)": r4,
        "Emissions (gCO2)": g4,
    },
)
state5_node = Node(
    "state5",
    0,
    properties={
        "EV Battery (kWh)": e5,
        "Reward ($)": r5,
        "Emissions (gCO2)": g5,
    },
)

e6 = round(e2 - dE, 2)
e7 = round(e2, 2)
g6 = round(1.84 * -dE, 2)
g7 = round(0, 2)
r6 = round(dE, 2)
r7 = round(0, 2)

state6_node = Node(
    "state6",
    0,
    properties={
        "EV Battery (kWh)": e6,
        "Reward ($)": r6,
        "Emissions (gCO2)": g6,
    },
)
state7_node = Node(
    "state7",
    0,
    properties={
        "EV Battery (kWh)": e7,
        "Reward ($)": r7,
        "Emissions (gCO2)": g7,
    },
)

e8 = round(e4 - dE, 2)
e9 = round(e4, 2)
g8 = round(1.84 * -dE, 2)
g9 = round(0, 2)
r8 = round(dE, 2)
r9 = round(0, 2)

state8_node = Node(
    "state8",
    0,
    properties={
        "EV Battery (kWh)": e8,
        "Reward ($)": r8,
        "Emissions (gCO2)": g8,
    },
)
state9_node = Node(
    "state9",
    0,
    properties={
        "EV Battery (kWh)": e9,
        "Reward ($)": r9,
        "Emissions (gCO2)": g9,
    },
)

tree.insert("initial state", "u1", [state1_node], [2.08], [1])
tree.insert("initial state", "u2", [state2_node], [0], [1])
# tree.insert("initial state", "u3", [state3_node], [-2.08], [1])

tree.insert("state1", "u1", [state4_node], [2.08], [1])
tree.insert("state1", "u2", [state5_node], [0], [1])

tree.insert("state2", "u1", [state6_node], [2.08], [1])
# tree.insert("state2", "u2", [state7_node], [0], [1])

tree.insert("state4", "u1", [state8_node], [2.08], [1])
tree.insert("state4", "u2", [state9_node], [0], [1])

tree_viz = tree.to_graphviz()
st.graphviz_chart(tree_viz, use_container_width=True)
