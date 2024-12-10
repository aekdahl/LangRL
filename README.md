# LangRL

LangRL is a Python package for creating dynamic workflows using **Reinforcement Learning (RL)** and **state-based graphs**. It allows you to define workflows as graphs, traverse them dynamically with RL agents, and adapt to real-time decision-making needs.

## Features

- **RL-Driven Traversal**: Dynamically determine the best routes in workflows using RL agents.
- **State-Based Graphs**: Define workflows with custom nodes and edges for fine-grained control.
- **Extensibility**: Works seamlessly with LangChain or as a standalone solution.
- **Fallback Support**: Includes a built-in graph framework for environments without LangChain.

## Installation

LangRL is available on PyPI. Install it using pip:

```bash
pip install langrl
```

## Getting Started

```python
# Define a Workflow
from langrl import RLControlledGraph

# Define a simple RL agent
class MyRLAgent:
    def decide(self, state, possible_edges):
        return possible_edges[0]  # Always choose the first edge (simple example)

# Initialize the graph and agent
graph = RLControlledGraph(rl_agent=MyRLAgent())

# Add nodes with their processing functions
graph.add_node("Start", lambda state: {"context": state["context"] + " -> Start"})
graph.add_node("Middle", lambda state: {"context": state["context"] + " -> Middle"})
graph.add_node("End", lambda state: {"context": state["context"] + " -> End"})

# Define edges between nodes
graph.add_edge("Start", "Middle")
graph.add_edge("Middle", "End")

# Traverse the graph
final_state = graph.invoke(start_node="Start", goal_node="End")
print(f"Final Context: {final_state['context']}")

# Output:
At Node: Start
At Node: Middle
At Node: End
Traversal completed. Final Context: Workflow -> Start -> Middle -> End
```

## Use Cases

- **Workflow Automation**: Automate dynamic workflows in business, research, and AI systems.
- **Game Development**: Create AI behavior trees with adaptive pathfinding.
- **Decision Systems**: Optimize processes like supply chain management or task prioritization.

## Contributing
LangRL is open source, and we welcome contributions! To get started:

1. Fork the repository.
2. Create a feature branch: <code>git checkout -b feature-name</code>.
3. Commit your changes: <code>git commit -m "Add feature"</code>.
4. Push to your branch: <code>git push origin feature-name</code>.
5. Open a pull request.

## License
LangRL is released under the MIT License. See LICENSE for details.

## Stay Connected
- **Issues**: Report bugs or request features in the issue tracker.
- **Discussions**: Join the conversation in the discussions tab.
