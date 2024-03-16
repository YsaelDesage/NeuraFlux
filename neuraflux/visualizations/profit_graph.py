import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np

def create_rl_graphs_updated(reward_data, profit_data, training_times):
    """
    Update the graph creation function to match the fill color with the line color and to make the reward increase
    abruptly after training times in the left plots.

    :param reward_data: A dict with keys as environment names and values as tuples of (baseline_rewards, agent_rewards)
                        where each is an array of shape (num_experiments, num_timesteps).
    :param profit_data: A dict similar to reward_data but for cumulative profits.
    :param training_times: A list of training times (t) for the vertical dashed lines.
    :return: Plotly figure containing the 4x2 graphs.
    """

    # Colors for each graph
    line_colors = ['rgba(255, 0, 0, 1)', 'rgba(0, 0, 255, 1)', 'rgba(255, 215, 0, 1)', 'rgba(0, 128, 0, 1)']
    fill_colors = ['rgba(255, 0, 0, 0.2)', 'rgba(0, 0, 255, 0.2)', 'rgba(255, 215, 0, 0.2)', 'rgba(0, 128, 0, 0.2)']

    # Create subplots with increased height
    fig = make_subplots(rows=4, cols=2, vertical_spacing=0.05, horizontal_spacing=0.03,subplot_titles=[
            "<b>Building HVAC Reward</b>", "<b>Building HVAC Profit</b>", 
            "<b>Energy Storage Reward</b>", "<b>Energy Storage Profit</b>", 
            "<b>Electric Vehicle Reward</b>", "<b>Electric Vehicle Profit</b>", 
            "<b>Industrial Process Reward</b>", "<b>Industrial Process Profit</b>"
        ])

    for i, (env_name, line_color, fill_color) in enumerate(zip(reward_data.keys(), line_colors, fill_colors)):
        # Rewards
        baseline_rewards, agent_rewards = reward_data[env_name]
        baseline_profit, agent_profit = profit_data[env_name]

        # Modify agent_rewards to increase abruptly after training times
        for t in training_times:
            agent_rewards[:, t:] += np.random.rand(num_experiments, num_timesteps - t) * 10
            agent_profit[:, t:] += np.random.rand(num_experiments, num_timesteps - t) * 10
            agent_profit[:, :] += np.arange(agent_profit.shape[1])

        fig.add_trace(go.Scatter(x=np.arange(baseline_rewards.shape[1]), y=baseline_rewards.mean(axis=0),
                                 line=dict(color='rgba(169, 169, 169, 1)'), name=f'{env_name} Baseline'),
                      row=i+1, col=1)

        fig.add_trace(go.Scatter(x=np.arange(agent_rewards.shape[1]), y=agent_rewards.max(axis=0),
                                 line=dict(color='rgba(0, 0, 0, 0)'), name=f'{env_name} Agent'),
                      row=i+1, col=1)

        fig.add_trace(go.Scatter(x=np.arange(agent_rewards.shape[1]), y=agent_rewards.min(axis=0),
                                 fill='tonexty', fillcolor=fill_color, line=dict(color='rgba(0, 0, 0, 0)'), name=f'{env_name} Agent'),
                      row=i+1, col=1)
        
        fig.add_trace(go.Scatter(x=np.arange(agent_rewards.shape[1]), y=agent_rewards.mean(axis=0),
                                 line=dict(color=line_color), name=f'{env_name} Agent'),
                      row=i+1, col=1)

        # Cumulative profits
        fig.add_trace(go.Scatter(x=np.arange(baseline_profit.shape[1]), y=baseline_profit.mean(axis=0),
                                 line=dict(color='rgba(169, 169, 169, 1)'), showlegend=False),
                      row=i+1, col=2)

        fig.add_trace(go.Scatter(x=np.arange(agent_profit.shape[1]), y=agent_profit.max(axis=0),
                                 line=dict(color='rgba(0, 0, 0, 0)'), showlegend=False),
                      row=i+1, col=2)
        fig.add_trace(go.Scatter(x=np.arange(agent_profit.shape[1]), y=agent_profit.min(axis=0),
                                 fill='tonexty', fillcolor=fill_color, line=dict(color='rgba(0, 0, 0, 0)'), showlegend=False),
                      row=i+1, col=2)
        fig.add_trace(go.Scatter(x=np.arange(agent_profit.shape[1]), y=agent_profit.mean(axis=0),
                                 line=dict(color=line_color), showlegend=False),
                      row=i+1, col=2)

        # Add vertical line for training time
        for t in training_times:
            fig.add_vline(x=t, line=dict(color='black', dash='dash'), row=i+1, col=1)
            fig.add_vline(x=t, line=dict(color='black', dash='dash'), row=i+1, col=2)

    # Update layout with increased height and white background
    fig.update_layout(height=1500, width=1200, title_text="RL Environments Performance", plot_bgcolor='white', showlegend=False)

    return fig

# Example data
num_environments = 4
num_experiments = 5
num_timesteps = 24 * 28
training_times = [24*7, 24*14, 24*21]

# Update the example data to reflect the abrupt increase in rewards after training times
np.random.seed(0)
reward_data_updated = {}
profit_data = {}
for i in range(num_environments):
    baseline_rewards = np.random.rand(num_experiments, num_timesteps) * 5 + 5
    agent_rewards = np.random.rand(num_experiments, num_timesteps) * 6 + 6
    reward_data_updated[f'Env{i+1}'] = (baseline_rewards, agent_rewards)
    profit_data[f'Env{i+1}'] = (np.cumsum(np.random.rand(num_experiments, num_timesteps) * 30, axis=1), 
                                np.cumsum(np.random.rand(num_experiments, num_timesteps) * 50, axis=1) + np.random.randint(-3000, 3000, (num_experiments,num_timesteps)))

# Create and display the graphs with the updated function
fig_updated = create_rl_graphs_updated(reward_data_updated, profit_data, training_times)
fig_updated.show()