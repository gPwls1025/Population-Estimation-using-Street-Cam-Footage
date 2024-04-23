import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from process import create_filtered_word_counts_df, update_co_occurrence
import plotly.express as px

# Define human actions
human_actions = ['drive','ride','cross','walk','pick up','stand','carry','catch','jog','spray','push','skate','wash','travel',
                 'clean','wear','crowded','take','run','swab','drag','play','check','stretch']
    

def create_graph_from_co_occurrence(co_occurrence_matrix, word_counts_df):
    G = nx.Graph()
    filtered_tags = set(word_counts_df['Tag'])  # Get only the tags present in word_counts_df_filtered
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
            G.nodes[node]['size'] = word_count[0]  # Taking the word count from word_counts_df
    return G

# Creating a Plotly figure
def plot_network(graph):
    pos = nx.spring_layout(graph, seed=42)  # Layout for better visualization

    max_weight = max([graph.edges[edge]['weight'] for edge in graph.edges()]) if graph.edges() else 1

    edge_trace = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = graph.edges[edge]['weight']
        edge_trace.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None], mode='lines', line=dict(width=0.01 * weight, color=f'rgb({int(128 * weight / max_weight)}, {int(128 * weight / max_weight)}, {int(128 * weight / max_weight)})')))  # Adjusting edge width and color

    node_trace = go.Scatter(x=[], y=[], mode='markers', text=[], marker=dict(size=[], color=[], colorscale='Viridis', opacity=0.7))
    for node in graph.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['marker']['size'] += (graph.nodes[node].get('size', 1) * 0.03,)  # Adjusting node size
        connected_nodes = list(graph.neighbors(node))
        top_connected_nodes = sorted(connected_nodes, key=lambda x: graph.edges[(node, x)]['weight'], reverse=True)[:20]  # Get top 20 most connected nodes
        top_connected_nodes_text = "<br>".join(top_connected_nodes) if top_connected_nodes else "None"
        #node_trace['text'] += (node,)
        node_trace['text'] += ([f'Node: {node}<br>Connections: {len(connected_nodes)}<br>Top Connected Nodes: {top_connected_nodes_text}'],)  # Adjusting node text
        node_trace['marker']['color'] += (graph.nodes[node].get('size', 1) * 10,)  # Adjusting node color

    fig = go.Figure(data=[*edge_trace, node_trace],
                    layout=go.Layout(title='Network Graph', showlegend=False, hovermode='closest', 
                                     margin=dict(b=20, l=5, r=5, t=40),
                                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

def get_tag_data_without_label(tag, *dfs):
    # Get data for a specific tag from multiple DataFrames, excluding 'Tag' label
    data = {}
    for df in dfs:
        if tag in df['Tag'].values:
            # Get the count from the DataFrame, setting to zero if not found
            data[df.name.capitalize()] = df.loc[df['Tag'] == tag, 'RAM_Count'].values[0]
        else:
            data[df.name.capitalize()] = 0
    return data
    
def plot_tag_counts_by_location(df, location):
    # Filter the data by the specified location
    df_filtered = df[df['location'] == location]
    
    # Get the filtered words
    filtered_words = ' | '.join(df_filtered['RAM_Tags']).split('|')
    filtered_words = [word.strip().lower() for word in filtered_words]
    
    # Count the occurrences of each word
    filtered_word_counts = pd.Series(filtered_words).value_counts()
    
    # Exclude words that occur too frequently in df
    threshold = 0.75 * len(df_filtered)
    filtered_word_counts = filtered_word_counts[filtered_word_counts <= threshold]
    #top_10_tags = filtered_word_counts.nlargest(10)
    
    fig = px.bar(filtered_word_counts, x=filtered_word_counts.index, y=filtered_word_counts.values)
    fig.update_layout(title=f'Tag Counts for Location: {location}', xaxis_title='Tag', yaxis_title='Count')
    fig.update_xaxes(tickangle=45)

    return fig