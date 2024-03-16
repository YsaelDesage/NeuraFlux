import random
import graphviz

class IndustrialProcessMDP:
    def __init__(self):
        self.states = ["Low Production", "High Production", "Maintenance", "Broken", "Idle"]
        self.actions = ["Increase Output", "Decrease Output", "Perform Maintenance", "Repair", "Shutdown"]
        self.transitions = self._init_transitions()
        self.energy_consumption = self._init_energy_consumption()
        self.rewards = self._init_rewards()
        self.current_state = None

    def _init_transitions(self):
        # Define transitions with probabilities
        transitions = {
            "Low Production": {
                "Increase Output": ("High Production", 1.0),
                "Decrease Output": ("Idle", 1.0),
                "Breakdown": ("Broken", 0.1),  # Probabilistic transition
            },
            "High Production": {
                "Decrease Output": ("Low Production", 1.0),
                "Breakdown": ("Broken", 0.2),  # Probabilistic transition
            },
            "Maintenance": {
                "Increase Output": ("Idle", 1.0),
            },
            "Broken": {
                "Repair": ("Maintenance", 1.0),
            },
            "Idle": {
                "Increase Output": ("Low Production", 1.0),
                "Maintenance": ("Maintenance", 1.0),  # Added deterministic transition
            }
        }
        return transitions

    def _init_energy_consumption(self):
        # Define energy consumption for each state
        return {
            "Low Production": 50,  # Example values
            "High Production": 100,
            "Maintenance": 20,
            "Broken": 0,
            "Idle": 10,
        }

    def _init_rewards(self):
        # Define rewards (money) for each transition
        return {
            ("Low Production", "Increase Output"): 200,
            ("High Production", "Decrease Output"): 150,
            # Continue for other state-action pairs
        }

    def initialize(self):
        # Initialize the system to a random state
        self.current_state = random.choice(self.states)

    def step(self, action):
        # Transition the system to another state based on the action and probabilities
        if action in self.transitions[self.current_state]:
            next_state, probability = self.transitions[self.current_state][action]
            if random.random() < probability:
                self.current_state = next_state
                reward = self.rewards.get((self.current_state, action), 0)
                return next_state, reward
            else:
                return self.current_state, 0  # No transition occurs
        else:
            return self.current_state, 0  # No transition and no reward

    def visualize_process(self):
        dot = graphviz.Digraph(comment='Industrial Process MDP', format='png')
        for state in self.states:
            dot.node(state, state)
            for action, (next_state, probability) in self.transitions[state].items():
                label = f"{action}"
                if probability < 1.0:
                    dot.edge(state, next_state, label=label, style="dotted")
                else:
                    dot.edge(state, next_state, label=label)
        return dot
    
    def simulate(self, steps=10):
        history = []
        self.initialize()
        for _ in range(steps):
            action = random.choice(self.actions)  # Randomly choose an action
            next_state, reward = self.step(action)
            history.append((self.current_state, action, next_state))
        return history

    def visualize_simulation(self, history):
        dot = graphviz.Digraph(comment='Industrial Process Simulation', format='png')

        # Create nodes for each state
        for state in self.states:
            dot.node(state, state)

        # Add edges for transitions in history
        for i, (state, action, next_state) in enumerate(history):
            label = f"{i}: {action}"
            dot.edge(state, next_state, label=label)

        return dot

# Example usage
mdp = IndustrialProcessMDP()
history = mdp.simulate(steps=10)

# Print the history for reference
print(history)

# # Example usage
# mdp = IndustrialProcessMDP()
# mdp.initialize()
# print(f"Initial State: {mdp.current_state}")
# next_state, reward = mdp.step("Increase Output")
# print(f"Next State: {next_state}, Reward: {reward}")

# # Visualization
process_graph = mdp.visualize_process()
process_graph.render(view=True)
