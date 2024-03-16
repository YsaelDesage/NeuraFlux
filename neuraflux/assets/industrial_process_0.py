import graphviz

class IndustrialProcessMDP:
    def __init__(self):
        self.states = ["Low Production", "High Production", "Maintenance", "Broken", "Idle"]
        self.actions = ["Increase Output", "Decrease Output", "Perform Maintenance", "Repair", "Shutdown"]
        self.transitions = self._init_transitions()

    def _init_transitions(self):
        # Define deterministic transitions for a simpler graph
        transitions = {
            "Low Production": {
                "Increase Output": "High Production",
                "Shutdown": "Idle",
            },
            "High Production": {
                "Decrease Output": "Low Production",
                "Shutdown": "Idle",
            },
            "Maintenance": {
                "Repair": "Low Production",
            },
            "Broken": {
                "Repair": "Maintenance",
            },
            "Idle": {
                "Increase Output": "Low Production",
            }
        }
        return transitions

    def visualize_process(self):
        dot = graphviz.Digraph(comment='Industrial Process MDP', format='png')
        for state in self.states:
            dot.node(state, state)
            for action, next_state in self.transitions[state].items():
                label = f"{action}"
                dot.edge(state, next_state, label=label)
        return dot

# Example usage
mdp = IndustrialProcessMDP()
process_graph = mdp.visualize_process()
process_graph.render(view=True)
