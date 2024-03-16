import pandas as pd
import plotly.graph_objects as go
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
    # Determine the overall min and max power values for a uniform colorscale
    min_power = min(df[power_column].min(), df[baseline_power_column].min())
    max_power = max(df[power_column].max(), df[baseline_power_column].max())

    # Create subplots with three rows
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "Temperature Variation",
            "AI Agent Power Consumption",
            "Baseline Power Consumption",
        ),
        row_heights=[0.6, 0.2, 0.2],
    )

    # Fill above the cooling setpoint with light grey
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=[100] * len(df),  # Upper boundary, set to a high value
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[cooling_setpoint_column],
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            fill="tonexty",
            fillcolor="rgba(211, 211, 211, 0.3)",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Fill below the heating setpoint with light grey
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=[-100] * len(df),  # Lower boundary, set to a low value
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[heating_setpoint_column],
            mode="lines",
            fillcolor="rgba(211, 211, 211, 0.3)",
            fill="tonexty",
            showlegend=False,
        ),
        row=1,
        col=1,
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
            y=df[cooling_setpoint_column],
            name="Cooling Setpoint",
            mode="lines",
            line=dict(color="red", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[heating_setpoint_column],
            name="Heating Setpoint",
            mode="lines",
            line=dict(color="blue", width=2),
        ),
        row=1,
        col=1,
    )

    fig.update_yaxes(range=[15, 29], row=1, col=1)

    # AI Agent Power Consumption Plot
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
    fig.update_yaxes(range=[0, max_power], row=2, col=1)

    # Baseline Power Consumption Plot
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df[baseline_power_column],
            name="Baseline Power",
            marker=dict(color=df[baseline_power_column], coloraxis="coloraxis"),
        ),
        row=3,
        col=1,
    )
    fig.update_yaxes(range=[0, max_power], row=3, col=1)

    # Update layout with a uniform colorscale for both power plots
    fig.update_layout(
        coloraxis=dict(
            colorscale="OrRd", cmin=min_power, cmax=max_power
        ),  # Uniform colorscale
        showlegend=False,
        plot_bgcolor="white",
    )
    # fig.update_xaxes(title_text="Time", row=1, col=1)
    # fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Power (kW)", row=2, col=1)
    fig.update_yaxes(title_text="Power (kW)", row=3, col=1)

    return fig


def plot_histograms(
    df: pd.DataFrame,
    columns: list[str],
    title="Expected Returns Distribution",
    xaxis_title="",
    yaxis_title="Count",
    barmode="overlay",
) -> go.Figure:
    """
    Plots overlaid histograms for the given columns in a DataFrame using Plotly.

    Parameters:
    - df: DataFrame containing the data
    - columns: List of column names for which histograms should be created
    - title: Title of the histogram (default is "Overlaid Histograms")
    - xaxis_title: Title of the x-axis (default is empty)
    - yaxis_title: Title of the y-axis (default is "Count")
    - barmode: Mode of histogram bars. It can be 'overlay' (default) or 'group'.
    """

    if not set(columns).issubset(df.columns):
        raise ValueError("Not all columns are present in the provided DataFrame")

    # Create the empty figure
    fig = go.Figure()

    # Add histograms for each column
    for col in columns:
        fig.add_trace(go.Histogram(x=df[col], name=col, opacity=0.6, nbinsx=len(df)))

    # Update the layout
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        barmode=barmode,
        plot_bgcolor="white",  # Setting the plot background to white
        paper_bgcolor="white",  # Setting the paper background to white
    )
    fig.update_traces(opacity=0.65)
    return fig


def plot_q_factor_bars(q_factors):
    titles = ["Control 1", "Control 2", "Control 3", "Control 4", "Control 5"][
        : len(q_factors)
    ]
    fig = make_subplots(rows=1, cols=len(q_factors))

    for idx, val in enumerate(q_factors, 1):
        # Color based on value
        color = "green" if val > 0 else "red"

        # Create a gradient background
        gradient = go.Bar(
            x=[0],
            y=[max(q_factors) + 1],
            marker=dict(
                color=[f"rgba(255, 0, 0, {abs(val)/max(q_factors)})"]
                if val < 0
                else [f"rgba(0, 255, 0, {abs(val)/max(q_factors)})"]
            ),
            hoverinfo="none",
        )

        # Create a bar trace
        bar = go.Bar(x=[0], y=[val], marker=dict(color=color), hoverinfo="none")

        fig.add_trace(gradient, row=1, col=idx)
        fig.add_trace(bar, row=1, col=idx)

        # Adjust the axis range
        fig.update_xaxes(range=[0, 1], row=1, col=idx)
        fig.update_yaxes(range=[min(q_factors) - 1, max(q_factors) + 1], row=1, col=idx)

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

    # Adding control titles under each bar with improved alignment
    for i, title in enumerate(titles, 1):
        fig.add_annotation(
            dict(
                font=dict(color="black", size=15, family="Arial, bold"),
                x=0.5,
                y=min(q_factors) - 0.5,
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
    )

    return fig


def plot_circle_subplot(q_factors, min_range=-1, max_range=1):
    if len(q_factors) not in [3, 5]:
        raise ValueError("The number of Q-factors should be 3 or 5.")

    titles = ["Control 1", "Control 2", "Control 3", "Control 4", "Control 5"][
        : len(q_factors)
    ]
    fig = make_subplots(rows=1, cols=len(q_factors))

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
            "text": [str(val)],
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
    )

    return fig


def _plot_circle_subplot(val1, val2, val3, min_range=-1, max_range=1):
    values = [val1, val2, val3]
    titles = ["Control 1", "Control 2", "Control 3"]
    fig = make_subplots(rows=1, cols=3)

    for idx, val in enumerate(values, 1):
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
            "text": [str(val)],
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
        # fig.update_layout(showlegend=False)

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
    )

    return fig
