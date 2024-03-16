import plotly.graph_objs as go
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots


def create_hourly_ghg_emissions_heatmap():
    # Dummy data for a month with hourly granularity
    days = pd.date_range(start="2023-01-01", end="2023-01-14", freq="D").strftime(
        "%Y-%m-%d"
    )
    hours = list(range(24))
    data_base = []
    data_optimized = []

    for day in days:
        for hour in hours:
            # Base scenario emissions for each hour
            base_emission = np.random.uniform(20, 50)
            data_base.append([day, hour, base_emission])

            # NeuraFlux optimized emissions for each hour
            optimized_emission = np.random.uniform(0, 14)
            data_optimized.append([day, hour, optimized_emission])

    # Creating DataFrames for each scenario
    df_base = pd.DataFrame(data_base, columns=["Day", "Hour", "Emissions"])
    df_optimized = pd.DataFrame(data_optimized, columns=["Day", "Hour", "Emissions"])

    # Define the shared colorscale based on the max emission value
    max_emission_value = max(
        df_base["Emissions"].max(), df_optimized["Emissions"].max()
    )
    scale_max = 50  # Assuming 50 as the maximum scale value for emissions

    # Defining the colorscale
    colorscale = [
        [0, "rgb(44, 123, 182)"],  # Cooler blue for lower emissions
        [0.2, "rgb(171, 217, 233)"],
        [0.4, "rgb(255, 255, 191)"],
        [0.6, "rgb(253, 174, 97)"],
        [0.8, "rgb(253, 141, 60)"],
        [1, "rgb(240, 59, 32)"],  # Warmer orange/red for higher emissions
    ]

    # Creating two subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Base Scenario", "NeuraFlux Optimized"),
        vertical_spacing=0.1,
    )

    # Base Scenario heatmap
    fig.add_trace(
        go.Heatmap(
            x=df_base["Hour"],
            y=df_base["Day"],
            z=df_base["Emissions"],
            coloraxis="coloraxis",
        ),
        row=1,
        col=1,
    )

    # NeuraFlux Optimized heatmap
    fig.add_trace(
        go.Heatmap(
            x=df_optimized["Hour"],
            y=df_optimized["Day"],
            z=df_optimized["Emissions"],
            coloraxis="coloraxis",
        ),
        row=2,
        col=1,
    )

    # Updating layout
    fig.update_layout(
        height=800,  # Making the figure taller to make heatmaps more square
        title="Hourly GHG Emissions Reduction Comparison for a Month",
        xaxis_nticks=24,  # To show each hour
        yaxis_nticks=14,  # To show each day of the month
        xaxis2_nticks=24,
        yaxis2_nticks=14,
        coloraxis={"colorscale": "Oranges"},
    )

    fig.update_layout(coloraxis={"colorscale": colorscale})

    return fig


# Example use of the function
fig = create_hourly_ghg_emissions_heatmap()
fig.show()
