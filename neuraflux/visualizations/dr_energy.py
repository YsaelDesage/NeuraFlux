import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

def create_ev_demand_response_plot(energy_profile, start_event, end_event):
    # Create a time series for the energy profile
    time_series = pd.date_range(start='2023-01-01', periods=len(energy_profile), freq='5T')
    df = pd.DataFrame({'Time': time_series, 'Energy Consumption': energy_profile})

    # Create the plot
    fig = go.Figure()

    # Add the energy consumption line
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Energy Consumption'], mode='lines', name='Energy Consumption'))

    # Highlight the demand response event
    fig.add_vrect(x0=start_event, x1=end_event, fillcolor="orange", opacity=0.5, line_width=0)

    # Update layout with white background
    fig.update_layout(title="EV Energy Consumption and Demand Response Event",
                      xaxis_title="Time",
                      yaxis_title="Energy Consumption (kWh)",
                      plot_bgcolor='white',
                      showlegend=True)

    return fig

# Realistic dummy data for testing
base_consumption = 1
energy_profile = [base_consumption + np.random.normal(0, 0.05) for _ in range(288)]  # Normal daily variation
start_event_idx = 168  # Corresponding to 14:00
end_event_idx = 216  # Corresponding to 18:00
for i in range(start_event_idx, end_event_idx):
    energy_profile[i] *= 0.5  # Simulate a decrease in consumption during the event

start_event = '2023-01-01 14:00'
end_event = '2023-01-01 18:00'

# Create the plot
fig = create_ev_demand_response_plot(energy_profile, start_event, end_event)

# Show the plot
fig.show()
