import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from process import process_data

# Define human actions
human_actions = ['drive','ride','cross','walk','pick up','stand','carry','catch','jog','spray','push','skate','wash','travel',
                 'clean','wear','crowded','take','run','swab','drag','play','check','stretch']
    

def create_graph_from_co_occurrence(co_occurrence_matrix, word_counts_df):
    # Call the process_data function
    #df, word_counts_df_filtered = process_data('/Users/hp/Population-Estimation-using-Street-Cam-Footage/YOLO+RAM_merged.csv')

    # create empty graph G
    G = nx.Graph()
    # retrieves set of indexes
    filtered_tags = set(word_counts_df['Tag'])  
    # iterate over rows and columns of co-occurence matrix, check if both tags are in 'filtered_tags'
    for i in range(len(co_occurrence_matrix.index)):
        tag1 = co_occurrence_matrix.index[i]
        if tag1 in filtered_tags:
            G.add_node(tag1) 
            for j in range(i + 1, len(co_occurrence_matrix.columns)):
                tag2 = co_occurrence_matrix.columns[j]
                if tag2 in filtered_tags:
                    weight = co_occurrence_matrix.iloc[i, j]
                    if weight >= 3:  # Only add edge if co-occurrence count is 3 or more
                        G.add_edge(tag1, tag2, weight=weight)
    # Assigning node size based on word counts
    for node in G.nodes():
        word_count = word_counts_df[word_counts_df['Tag'] == node]['RAM_Count'].values
        if len(word_count) > 0:
            G.nodes[node]['size'] = word_count[0] 
    return G

def plot_network(graph, title):
    # Call the process_data function
    #df, word_counts_df_filtered = process_data('/Users/hp/Population-Estimation-using-Street-Cam-Footage/YOLO+RAM_merged.csv')
    pos = nx.spring_layout(graph, seed=42)  # Layout for better visualization

    max_weight = max([graph.edges[edge]['weight'] for edge in graph.edges()]) if graph.edges() else 1

    # Define color mapping for human action tags
    color_mapping = {}
    for node in graph.nodes():
        if node in human_actions:
            color_mapping[node] = 'orange'  # color for human action tags
        else:
            color_mapping[node] = 'gray'  # Default color for non-human action tags

    edge_trace = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = graph.edges[edge]['weight']
        # Get color based on the tags of the connected nodes
        color = 'gray'  # Default color
        if edge[0] in color_mapping and edge[1] in color_mapping:
            color = 'orange' if color_mapping[edge[0]] == 'orange' or color_mapping[edge[1]] == 'orange' else 'gray'
        
        edge_trace.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None], mode='lines', 
                                     line=dict(width=0.04 * weight, color=color)))

    node_trace = go.Scatter(x=[], y=[], mode='markers', text=[], marker=dict(size=[], color=[], colorscale='Viridis', opacity=0.7))
    for node in graph.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['marker']['size'] += (graph.nodes[node].get('size', 1) * 0.04,)  # Adjusting node size
        connected_nodes = list(graph.neighbors(node))
        top_connected_nodes = sorted(connected_nodes, key=lambda x: graph.edges[(node, x)]['weight'], reverse=True)[:20]  # Get top 20 most connected nodes
        top_connected_nodes_text = ", ".join(top_connected_nodes) if top_connected_nodes else "None"
        node_trace['text'] += ([f'Node: {node}<br>Connections: {len(connected_nodes)}<br>Top Connected Nodes: {top_connected_nodes_text}'],)  # Adjusting node text
        node_trace['marker']['color'] += (graph.nodes[node].get('size', 1) * 10,)  # Adjusting node color

    fig = go.Figure(data=[*edge_trace, node_trace],
                    layout=go.Layout(title=title, showlegend=False, hovermode='closest',
                                     margin=dict(b=20, l=5, r=5, t=40),
                                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    # Adjust the size of the graph
    fig.update_layout(width=1200, height=900)


    fig.show()
    return fig
