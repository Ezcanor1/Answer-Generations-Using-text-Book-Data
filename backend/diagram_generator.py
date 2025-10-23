from graphviz import Digraph

def generate_flowchart(nodes, filename):
    dot = Digraph()
    for i, node in enumerate(nodes):
        dot.node(str(i), node)
    for i in range(len(nodes) - 1):
        dot.edge(str(i), str(i+1))
    dot.render(filename, format='png')

