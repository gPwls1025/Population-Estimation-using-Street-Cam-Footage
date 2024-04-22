import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from dash import Dash, html, dcc, callback, Input, Output
from plotly.graph_objects import Figure

# Assuming these are functions from your earlier code snippet
# - create_graph_from_co_occurrence
# - plot_network

def update_co_occurrence(df, co_occurrence_matrix):
    for _, row in df.iterrows():
        tags = row['RAM_Tags'].split(' | ')
        for i in range(len(tags)):
            for j in range(i+1, len(tags)):
                co_occurrence_matrix.at[tags[i], tags[j]] += 1
                co_occurrence_matrix.at[tags[j], tags[i]] += 1


def create_filtered_word_counts_df(df, location_id):
    filtered_df = df[df['locationID'] == location_id]
    text = ' | '.join(filtered_df['RAM_Tags'])
    words = text.split('|')
    words = [word.strip().lower() for word in words]
    word_counts = pd.Series(words).value_counts()
    
    word_counts_df = pd.DataFrame(word_counts)
    word_counts_df.reset_index(inplace=True)
    word_counts_df.columns = ['Tag', 'RAM_Count']
    word_counts_df_filtered = word_counts_df[~((word_counts_df['RAM_Count'] > 0.75 * len(filtered_df)) | (word_counts_df['RAM_Count'] <= 4))]
    
    # Calculate normalized counts
    normalized_counts = pd.DataFrame({'Tag': word_counts_df_filtered['Tag'], 'RAM_Count': word_counts_df_filtered['RAM_Count'] / len(filtered_df)})
    
    return word_counts_df_filtered, normalized_counts


def get_word_counts_and_co_occurrence(df):
    text = ' | '.join(df['RAM_Tags'])
    words = text.split('|')
    words = [word.strip().lower() for word in words]

    # Count the occurrences of each word
    word_counts = pd.Series(words).value_counts()

    #Make a df for storage
    word_counts_df = pd.DataFrame(word_counts)
    word_counts_df.reset_index(inplace=True)
    word_counts_df.columns = ['Tag','RAM_Count']

    all_tags = set()
    for tags in df['RAM_Tags']:
        all_tags.update(tags.split(' | '))
    all_tags = sorted(list(all_tags))

    # Creating co-occurrence matrices
    co_occurrence_overall = pd.DataFrame(index=all_tags, columns=all_tags).fillna(0)

    # Updating co-occurrence matrices
    update_co_occurrence(df, co_occurrence_overall) #This makes the co-occurence matrix

    #word_counts_df_filtered = word_counts_df[~((word_counts_df['RAM_Count']>0.75*len(df)) | (word_counts_df['RAM_Count'] <= 4))]

    # Assuming df is your DataFrame
    # Call the function for each locationID
    words_count_1, words_cps_1 = create_filtered_word_counts_df(df, 1)
    words_count_2, words_cps_2 = create_filtered_word_counts_df(df, 2)
    words_count_3, words_cps_3 = create_filtered_word_counts_df(df, 3)

    words_count_1['RAM_Count'] *= (1/3)
    words_count_2['RAM_Count'] *= (1/3)
    words_count_3['RAM_Count'] *= (1/3)

    merged_df_location = pd.concat([words_count_1, words_count_2, words_count_3], ignore_index=True)
    grouped_df_location = merged_df_location.groupby('Tag')['RAM_Count'].sum().reset_index()

    return grouped_df_location, co_occurrence_overall, words_cps_1, words_cps_2, words_cps_3



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

'''
def get_tag_data(tag, *dfs):
    data = {'Tag': tag}
    for df in dfs:
        if tag in df['Tag'].values:
            data[df.name.capitalize()] = df.loc[df['Tag'] == tag, 'RAM_Count'].values[0]
        else:
            data[df.name.capitalize()] = 0
    return data
'''

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

# Data processing logic for co-occurrence graph
df = pd.read_csv('YOLO+RAM_merged.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)

# Create a co-occurrence matrix and word counts
word_counts_df_filtered, co_occurrence_overall, words_cps_1, words_cps_2, words_cps_3 = get_word_counts_and_co_occurrence(df)

words_cps_1.name = 'park'
words_cps_2.name = 'chase'
words_cps_3.name = 'dumbo'

# Create the co-occurrence graph
graph = create_graph_from_co_occurrence(co_occurrence_overall, word_counts_df_filtered)


'''
# Create Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Graph(
            id='network-graph',
            figure=plot_network(graph)
        )
    ], style={'width': '100%', 'display': 'inline-block', 'padding': '0 20'})
])

if __name__ == '__main__':
    app.run(debug=True)
'''

# Create Dash app
app = Dash(__name__)

# Layout with network graph and bar chart
app.layout = html.Div([
    html.H1("Interactive Network Graph with Hover-based Bar Chart"),
    html.Div([
        dcc.Graph(
            id='network-graph',
            figure=plot_network(graph),
            hoverData={'points': [{'text': 'example_tag'}]}  # Default hover data
        )
    ], style={'width': '70%', 'display': 'inline-block'}),

    html.Div([
        dcc.Graph(id='bar-graph')  # Placeholder for the bar chart
    ], style={'width': '30%', 'display': 'inline-block'})
])

# Callback to update the bar chart based on hover data from the network graph
@callback(
    Output('bar-graph', 'figure'),
    [Input('network-graph', 'hoverData')]
)

def update_bar_chart(hover_data):
    tag = hover_data['points'][0]['text'].split("<br>")[0].replace("Node: ", "").strip("['") if hover_data and 'points' in hover_data else None
    
    if tag:
        # Get data for the tag
        tag_data = get_tag_data_without_label(tag, words_cps_1, words_cps_2, words_cps_3)

        # Create the bar plot with Plotly
        bar_fig = go.Figure(
            data=[
                go.Bar(x=list(tag_data.keys()), y=list(tag_data.values()), marker=dict(color=['red', 'green', 'blue']))
            ],
            layout=go.Layout(
                title=f'RAM Count for Tag: {tag}',
                xaxis=dict(title='Location'),
                yaxis=dict(title='RAM Count'),
                showlegend=False
            )
        )
        return bar_fig
    else:
        return go.Figure()  # Empty figure if no tag is hovered over

if __name__ == '__main__':
    app.run(debug=True)
