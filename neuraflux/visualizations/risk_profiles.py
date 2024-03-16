import plotly.graph_objects as go
import numpy as np
import pandas as pd

def create_profit_plot():
    # Time range for a month (assuming 30 days, 24 hours each)
    time = pd.date_range(start="2023-01-01", periods=30*24, freq='H')

    # Generating dummy data for three risk levels
    # Low risk: small positive slope, low variance
    low_risk_min = np.linspace(0, 200, len(time)) - np.random.uniform(5, 15, len(time))
    low_risk_max = np.linspace(0, 200, len(time)) + np.random.uniform(5, 15, len(time))
    low_risk_mean = (low_risk_min + low_risk_max) / 2

    # Medium risk: moderate slope, moderate variance
    med_risk_min = np.linspace(0, 300, len(time)) - np.random.uniform(10, 30, len(time))
    med_risk_max = np.linspace(0, 300, len(time)) + np.random.uniform(10, 30, len(time))
    med_risk_mean = (med_risk_min + med_risk_max) / 2

    # High risk: high slope, high variance
    high_risk_min = np.linspace(0, 400, len(time)) - np.random.uniform(20, 60, len(time))
    high_risk_max = np.linspace(0, 400, len(time)) + np.random.uniform(20, 60, len(time))
    high_risk_mean = (high_risk_min + high_risk_max) / 2

    # Creating the plot
    fig = go.Figure()

    # Adding the mean lines for each risk level
    fig.add_trace(go.Scatter(x=time, y=low_risk_mean, mode='lines', name='Low Risk', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=time, y=med_risk_mean, mode='lines', name='Medium Risk', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=time, y=high_risk_mean, mode='lines', name='High Risk', line=dict(color='red')))

    # Adding the min/max regions without legend entries
    fig.add_trace(go.Scatter(x=time, y=low_risk_min, mode='lines', name='Low Risk Min', showlegend=False, line=dict(width=0)))
    fig.add_trace(go.Scatter(x=time, y=low_risk_max, mode='lines', name='Low Risk Max', fill='tonexty', showlegend=False, line=dict(width=0), fillcolor='rgba(0,0,255,0.2)'))
    fig.add_trace(go.Scatter(x=time, y=med_risk_min, mode='lines', name='Medium Risk Min', showlegend=False, line=dict(width=0)))
    fig.add_trace(go.Scatter(x=time, y=med_risk_max, mode='lines', name='Medium Risk Max', fill='tonexty', showlegend=False, line=dict(width=0), fillcolor='rgba(255,165,0,0.2)'))
    fig.add_trace(go.Scatter(x=time, y=high_risk_min, mode='lines', name='High Risk Min', showlegend=False, line=dict(width=0)))
    fig.add_trace(go.Scatter(x=time, y=high_risk_max, mode='lines', name='High Risk Max', fill='tonexty', showlegend=False, line=dict(width=0), fillcolor='rgba(255,0,0,0.2)'))

    # Update layout
    fig.update_layout(title='Cumulative Hourly Profit Over Time for Different Risk Levels', xaxis_title='Time', yaxis_title='Cumulative Profit ($)', plot_bgcolor='white', legend=dict(x=0.1, y=0.95, font=dict(
                size=14  # Increase font size for larger legend text
            )))

    return fig

# To display the plot, you can call the function like this:
profit_plot = create_profit_plot()
profit_plot.show()
