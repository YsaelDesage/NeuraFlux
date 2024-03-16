import pandas as pd
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots


def plot_temperature_power(
    df,
    temp_column,
    baseline_temp_column,
    heating_setpoint_column,
    cooling_setpoint_column,
    power_column,
    baseline_power_column,
):
    # Create subplots with two rows
    fig = make_subplots(
        rows=2, cols=1, subplot_titles=("Temperature Variation", "Power Consumption")
    )

    # Temperature Variation Plot
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[temp_column],
            name="AI Agent Temperature",
            line=dict(color="black"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[baseline_temp_column],
            name="Baseline Temperature",
            line=dict(color="grey"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[heating_setpoint_column],
            name="Heating Setpoint",
            fill=None,
            mode="none",
            line=dict(color="blue", width=0),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[cooling_setpoint_column],
            name="Cooling Setpoint",
            fill="tonexty",
            mode="none",
            line=dict(color="red", width=0),
        ),
        row=1,
        col=1,
    )

    # Power Consumption Plot
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df[power_column],
            name="AI Agent Power",
            marker=dict(color=df[power_column], coloraxis="coloraxis"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df[baseline_power_column],
            name="Baseline Power",
            marker=dict(color=df[baseline_power_column], coloraxis="coloraxis"),
        ),
        row=2,
        col=1,
    )

    # Update layout
    fig.update_layout(
        coloraxis=dict(colorscale="YlOrRd"), showlegend=False, plot_bgcolor="white"
    )
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Power Consumption (kW)", row=2, col=1)

    return fig


def plot_q_factors_as_bars(q_factors, time_input, fixed_min=0, fixed_max=1):
    """
    Plot a bar graph of Q-factors using Plotly with improved aesthetics.

    Parameters:
    q_factors (list): A list of Q-factor values.
    fixed_min (float): Fixed minimum value for the y-axis.
    fixed_max (float): Fixed maximum value for the y-axis.
    """
    # Assigning colors based on the value and intensity of Q-factors
    max_q = max(q_factors)
    min_q = min(q_factors)

    colors = [
        f"rgba(0, 128, 34, {0.5 + 0.5 * (q / max_q)})"
        if q > 0
        else f"rgba(140, 3, 3, {0.5 + 0.5 * (abs(q) / abs(min_q))})"
        for q in q_factors
    ]

    # Creating the bar graph
    fig = go.Figure(
        data=[go.Bar(x=list(range(len(q_factors))), y=q_factors, marker_color=colors)]
    )

    # Updating layout for a cleaner white background
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        title={
            "text": time_input.strftime("%H:%M"),
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        margin=dict(t=40, b=0, l=0, r=0),  # Adjust top margin to make space for title,
        height=200,
    )
    fig.update_yaxes(
        showgrid=False,
        range=[fixed_min, fixed_max]
        if fixed_min is not None and fixed_max is not None
        else None,
    )

    return fig


def plot_q_factors_as_circles(
    q_factors,
    min_range=-1,
    max_range=1,
    titles=["Control 1", "Control 2", "Control 3", "Control 4", "Control 5"],
):
    if len(q_factors) not in [3, 5]:
        raise ValueError("The number of Q-factors should be 3 or 5.")

    titles = titles[: len(q_factors)]
    fig = make_subplots(rows=1, cols=len(q_factors))

    # Normalize Q-factors
    # q_factors = [round(q / sum(np.abs(q_factors)), 2) for q in q_factors]
    q_factors = np.round(np.tanh(np.array(q_factors) / 10000), 1)

    for idx, val in enumerate(q_factors, 1):
        # Color based on value
        color = "green" if val > 0 else "red"

        # Circle dimension proportional to absolute value
        size = abs(val)

        # Circle trace
        circle = {
            "type": "scatter",
            "x": [0],
            "y": [0],
            "mode": "markers+text",
            "text": [str(val * 100) + "%"],
            "textposition": "top center",
            "textfont": {"size": 18, "color": "black"},
            "marker": {
                "size": [
                    size * 150
                ],  # Multiply by a constant to make the circle visible
                "color": color,
            },
            "hoverinfo": "none",
        }

        fig.add_trace(circle, row=1, col=idx)

        # Adjust the axis range
        fig.update_xaxes(range=[min_range, max_range], row=1, col=idx)
        fig.update_yaxes(range=[min_range / 2, max_range / 2], row=1, col=idx)

        fig.update_xaxes(
            visible=False,
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            row=1,
            col=idx,
        )
        fig.update_yaxes(
            visible=False,
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            row=1,
            col=idx,
        )
        fig.update_layout(showlegend=False)

    # Adding control titles under each circle
    for i, title in enumerate(titles, 1):
        fig.add_annotation(
            dict(
                font=dict(color="black", size=15, family="Arial, bold"),
                x=0,
                y=-0.2,
                showarrow=False,
                text=title,
                xref="x" + str(i),
                yref="y" + str(i),
                xanchor="center",
                yanchor="bottom",
            )
        )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=0, b=0, l=0, r=0),
        height=450,
    )

    return fig


def create_financial_performance_plot(
    df, agent_cost_col, shadow_cost_col, color="red", reward: bool = False
):
    """
    Create a Plotly figure to show the performance of a Reinforcement Learning agent.

    Args:
    df (pd.DataFrame): DataFrame with a datetime index (5-minute intervals) and relevant columns.
    agent_cost_col (str): Column name for the agent's reward.
    shadow_cost_col (str): Column name for the default behavior's reward for comparison.

    Returns:
    plotly.graph_objs._figure.Figure: The Plotly figure object with the performance plots.
    """

    df = df.copy()

    # If building, calculate power contribution
    if "temperature_1" in df.columns:
        # Calculate the cumulative maximum power demand for both agent and shadow
        df["cummax_power"] = df["power"].cummax()
        df["cummax_shadow_power"] = df["shadow_power"].cummax()

        # Initialize demand charge columns
        df["demand_charge"] = 0
        df["shadow_demand_charge"] = 0

        # Apply demand charge only when there is a change in the max power demand
        df.loc[df["cummax_power"] > df["cummax_power"].shift(1), "demand_charge"] = (
            16.139 * df["cummax_power"]
        )
        df.loc[
            df["cummax_shadow_power"] > df["cummax_shadow_power"].shift(1),
            "shadow_demand_charge",
        ] = 16.139 * df["cummax_shadow_power"]

        # Update the agent and shadow cost columns to include the demand charge
        df[agent_cost_col] += df["demand_charge"]
        df[shadow_cost_col] += df["shadow_demand_charge"]

    # Calculate rolling averages over 12 periods for cost columns
    df["agent_cost_col_rolling_avg"] = df[agent_cost_col].rolling(window=12).mean()
    df["shadow_cost_col_rolling_avg"] = df[shadow_cost_col].rolling(window=12).mean()

    # Add cumulative profit column
    profit_col = "profit"
    if reward:
        df[profit_col] = df[agent_cost_col] - df[shadow_cost_col]
    else:
        df[profit_col] = df[shadow_cost_col] - df[agent_cost_col]

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            # "Average Cost Incurred",
            # "Agent Deployment Cumulative Cost Savings",
        ),
        row_heights=[0.7, 0.3],
    )

    # Hourly moving average of rewards
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["shadow_cost_col_rolling_avg"],
            mode="lines",
            name="Default Behavior",
            line=dict(color="grey"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["agent_cost_col_rolling_avg"],
            mode="lines",
            name="Agent Reward",
            line=dict(color=color),
        ),
        row=1,
        col=1,
    )

    # Cumulative profit
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[profit_col].cumsum(),
            mode="lines",
            # name="Cumulative Profit",
            fill="tozeroy",
            line=dict(color=color),
        ),
        row=2,
        col=1,
    )

    # Update layout
    fig.update_layout(template="plotly_white", showlegend=False)
    fig.update_xaxes(title_text="Time", row=2, col=1, showgrid=False)
    fig.update_yaxes(title_text="Cost ($)", row=1, col=1, showgrid=False)
    fig.update_yaxes(title_text="Agent Profit ($)", row=2, col=1, showgrid=False)

    return fig
