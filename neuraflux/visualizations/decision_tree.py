import graphviz
import numpy as np

def generate_detailed_decision_graph():
    # Define operations and transitions for each sequence
    operations = [f"Op{i}" for i in range(1, 6)]
    sequences = {
        "Seq1": [("Op1", "Op2"), ("Op2", "Op4"), ("Op4", "Op5")],
        "Seq2": [("Op1", "Op3"), ("Op3", "Op4"), ("Op4", "Op5")]
    }

    # Generate random metrics for each operation
    ghg_emissions = {op: np.random.randint(20, 100) for op in operations}
    profits = {op: np.random.randint(100, 500) for op in operations}
    energy_consumed = {op: np.random.randint(50, 200) for op in operations}

    # Calculate total metrics for each sequence
    total_metrics = {
        seq: {
            "Total Profit": sum(profits[op] for op, _ in transitions),
            "Total GHG": sum(ghg_emissions[op] for op, _ in transitions),
            "Total Energy": sum(energy_consumed[op] for op, _ in transitions)
        }
        for seq, transitions in sequences.items()
    }

    # Create a directed graph
    dot = graphviz.Digraph(comment='Detailed Operational Decision Graph')

    # Add an initial state node
    dot.node("Initial", "Initial State", shape="ellipse")
    colors = {"Seq1": "blue", "Seq2": "green"}
    # Add summary nodes for sequences with total metrics
    for seq, metrics in total_metrics.items():
        label = (f"{seq}\n"
                 f"Profit: {metrics['Total Profit']}\n"
                 f"GHG: {metrics['Total GHG']}\n"
                 f"Energy: {metrics['Total Energy']} \n"
                 f"Color: {colors[seq]}")
        dot.node(seq, label=label, shape="box", style="filled", color="lightgrey")

    # Add nodes and edges for each sequence with different colors
    for seq, transitions in sequences.items():
        color = colors[seq]
        for start, end in transitions:
            dot.node(start, f"{start}\nGHG: {ghg_emissions[start]}\nProfit: {profits[start]}\nEnergy: {energy_consumed[start]}")
            dot.node(end, f"{end}\nGHG: {ghg_emissions[end]}\nProfit: {profits[end]}\nEnergy: {energy_consumed[end]}")
            dot.edge(start, end, color=color)
        dot.edge("Initial", transitions[0][0], color=color)  # Edge from initial state

    return dot

# Generate and display the graph
detailed_decision_graph = generate_detailed_decision_graph()
detailed_decision_graph.render(view=True)