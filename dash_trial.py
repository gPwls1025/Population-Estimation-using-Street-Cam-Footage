import pandas as pd
import plotly.graph_objects as go
import networkx as nx
import json
from dash import Dash, html, dcc, callback, Input, Output
from plotly.graph_objects import Figure


def get_counts_df(df):
    text = ' | '.join(df['RAM_Tags'])
    words = text.split('|')
    words = [word.strip().lower() for word in words]
    word_counts = pd.Series(words).value_counts()
    word_counts_df = pd.DataFrame(word_counts)
    word_counts_df.reset_index(inplace=True)
    word_counts_df.columns = ['Tag', 'RAM_Count']
    return word_counts_df


def update_co_occurrence(df, co_occurrence_matrix):
    for _, row in df.iterrows():
        tags = row['RAM_Tags'].split(' | ')
        for i in range(len(tags)):
            for j in range(i+1, len(tags)):
                co_occurrence_matrix.at[tags[i], tags[j]] += 1
                co_occurrence_matrix.at[tags[j], tags[i]] += 1


def get_co_occurrence(df):
    all_tags = set()
    for tags in df['RAM_Tags']:
        all_tags.update(tags.split(' | '))
    all_tags = sorted(list(all_tags))

    # Creating co-occurrence matrices
    co_occurrence_matrix = pd.DataFrame(index=all_tags, columns=all_tags).fillna(0)
    update_co_occurrence(df, co_occurrence_matrix)
    return co_occurrence_matrix


def create_location_filtered_word_counts_df(df, location_id):
    filtered_df = df[df['locationID'] == location_id]
    word_counts_df = get_counts_df(filtered_df)
    word_counts_df_filtered = word_counts_df[~((word_counts_df['RAM_Count'] > 0.75 * len(filtered_df)) | (word_counts_df['RAM_Count'] <= 4))]
    normalized_counts = pd.DataFrame({'Tag': word_counts_df_filtered['Tag'], 'RAM_Count': word_counts_df_filtered['RAM_Count'] / len(filtered_df)})
    return word_counts_df_filtered, normalized_counts


def create_individual_location_data(df):
    # Call the function for each locationID
    words_count_1, words_cps_1 = create_location_filtered_word_counts_df(df, 1)
    words_count_2, words_cps_2 = create_location_filtered_word_counts_df(df, 2)
    words_count_3, words_cps_3 = create_location_filtered_word_counts_df(df, 3)

    words_cps_1.name = 'park'
    words_cps_2.name = 'chase'
    words_cps_3.name = 'dumbo'

    return words_count_1, words_cps_1, words_count_2, words_cps_2, words_count_3, words_cps_3


def create_aggregated_data(words_count_1, words_count_2, words_count_3):
    #Merge the dataframes
    words_count_1['RAM_Count'] *= (1/3)
    words_count_2['RAM_Count'] *= (1/3)
    words_count_3['RAM_Count'] *= (1/3)
    merged_df_location = pd.concat([words_count_1, words_count_2, words_count_3], ignore_index=True)
    grouped_df_location = merged_df_location.groupby('Tag')['RAM_Count'].sum().reset_index()

    return grouped_df_location


def create_gender_df(df):
    df['man'] = df['RAM_Tags'].str.contains(r'\bman\b|\bboy\b', case=False, regex=True).astype(int)
    df['woman'] = df['RAM_Tags'].str.contains(r'\bwoman\b|\bgirl\b', case=False, regex=True).astype(int)
    man_df = df[(df['man'] == 1) & (df['woman']==0)]
    woman_df = df[(df['man'] == 0) & (df['woman']==1)]
    return man_df, woman_df


def create_gender_filtered_word_counts_df(df):
    word_counts_df = get_counts_df(df)
    word_counts_df_filtered = word_counts_df[~((word_counts_df['RAM_Count'] <= 4))]
    normalized_counts = pd.DataFrame({'Tag': word_counts_df_filtered['Tag'], 'RAM_Count': word_counts_df_filtered['RAM_Count'] / len(df)})
    return normalized_counts


def create_gender_data(df):    
    #Call the function for the genders
    man_df, woman_df = create_gender_df(df)
    words_cps_m = create_gender_filtered_word_counts_df(man_df)
    words_cps_f = create_gender_filtered_word_counts_df(woman_df)
    
    words_cps_m.name = 'man'
    words_cps_f.name = 'woman'

    return words_cps_m, words_cps_f


def category_mapping(df):
    category_mapping = {'street corner': 'Urban Infrastructure','car': 'Vehicle','intersection': 'Urban Infrastructure',
                        'city street': 'Urban Infrastructure', 'cross': 'HAR', 'person': 'People', 'walk': 'HAR', 'pavement': 'Urban Infrastructure', 
                        'man': 'People', 'curb': 'Urban Infrastructure', 'crack': 'Urban Infrastructure', 'drain': 'Urban Infrastructure', 
                        'pedestrian': 'People', 'woman': 'People', 'manhole cover': 'Urban Infrastructure', 'drive': 'HAR', 
                        'manhole': 'Urban Infrastructure', 'street sign': 'Urban Infrastructure', 'city': 'Urban Infrastructure', 
                        'building': 'Urban Infrastructure', 'suv': 'Vehicle', 'ride': 'HAR', 'park': 'Vehicle', 'sedan': 'Vehicle', 
                        'zebra crossing': 'Urban Infrastructure', 'white': 'Color', 'sea': 'Miscellaneous', 'traffic sign': 'Urban Infrastructure', 
                        'stand': 'HAR', 'license plate': 'Vehicle', 'girl': 'People', 'skateboarder': 'People', 'biker': 'People', 
                        'vehicle': 'Vehicle', 'scooter': 'Vehicle', 'bicycle': 'Vehicle', 'black': 'Color', 'jog': 'HAR', 
                        'shirt': 'Clothing Accessory', 'spray': 'Personal Item', 'shopping bag': 'Personal Item', 
                        'blue': 'Color', 'van': 'Vehicle', 'motorcycle': 'Vehicle', 'minivan': 'Vehicle', 'dress': 'Clothing Accessory', 
                        'red': 'Color', 'bike lane': 'Urban Infrastructure', 'push': 'HAR', 'silver': 'Color', 'child': 'People', 
                        'bus': 'Vehicle', 'motorbike': 'Vehicle', 'traffic light': 'Urban Infrastructure', 'boy': 'People', 'couple': 'People', 
                        'motorcyclist': 'People', 'skate': 'HAR', 'skateboard': 'Personal Item', 'city bus': 'Vehicle', 'catch': 'HAR', 
                        'bag': 'Personal Item', 'sign': 'Urban Infrastructure', 'moped': 'Vehicle', 'baby carriage': 'Personal Item', 
                        'briefcase': 'Personal Item', 'carry': 'HAR', 'leash': 'Personal Item', 'puddle': 'Urban Infrastructure', 
                        'construction worker': 'People', 'chessboard': 'Personal Item', 'yellow': 'Color', 'truck': 'Vehicle', 
                        'ambulance': 'Vehicle', 'cane': 'Personal Item', 'taxi': 'Vehicle', 'wash': 'HAR', 'green': 'Color', 
                        'sandal': 'Clothing Accessory', 'pink': 'Color', 'office building': 'Urban Infrastructure', 'dog': 'Animal', 
                        'umbrella': 'Personal Item', 'tour bus': 'Vehicle', 'hand': 'Body Part', 'hose': 'Personal Item', 
                        'bicycle helmet': 'Personal Item', 'travel': 'HAR', 'phone': 'Personal Item', 'pick up': 'HAR', 
                        'bus stop': 'Vehicle', 'pole': 'Urban Infrastructure', 'chess': 'Miscellaneous', 'short': 'Clothing Accessory', 
                        'clean': 'HAR', 'cart': 'Personal Item', 'jeep': 'Vehicle', 'sports car': 'Vehicle', 'wear': 'Clothing Accessory', 
                        'business suit': 'Clothing Accessory', 'police': 'People', 'school bus': 'Vehicle', 'baby': 'People', 'smartphone': 
                        'Personal Item', 'officer': 'People', 'backpack': 'Personal Item', 'urban': 'Urban Infrastructure', 
                        'tie': 'Clothing Accessory', 'marathon': 'HAR', 'crowded': 'Miscellaneous', 'legging': 'Clothing Accessory', 
                        'take': 'HAR', 'trailer truck': 'Vehicle', 'paper bag': 'Personal Item', 'direct': 'HAR', 'khaki': 'Clothing Accessory', 
                        'shoe': 'Clothing Accessory', 'jeans': 'Clothing Accessory', 'limo': 'Vehicle', 'jacket': 'Clothing Accessory', 
                        'mobility scooter': 'Vehicle', 'broom': 'Personal Item', 'luggage': 'Personal Item', 'curve': 'Miscellaneous', 
                        'stop light': 'Urban Infrastructure', 'baseball hat': 'Personal Item', 'doodle': 'HAR', 'run': 'HAR', 'blanket': 
                        'Personal Item', 'head': 'Body Part', 'shirtless': 'Clothing Accessory', 'bulletin board': 'Personal Item', 
                        'dress shirt': 'Clothing Accessory', 'fire truck': 'Vehicle', 'police car': 'Vehicle', 'swab': 'HAR', 
                        'photo': 'HAR', 'purple': 'Color', 'pet': 'Animal', 'garbage truck': 'Vehicle', 'drag': 'HAR', 
                        'roller skate': 'Personal Item', 'shopping cart': 'Personal Item', 'monocycle': 'Vehicle', 'sweatshirt': 'Clothing Accessory', 
                        'protester': 'People', 'ponytail': 'Body Part', 'play': 'HAR', 'food truck': 'Vehicle', 'check': 'Miscellaneous', 
                        'tricycle': 'Vehicle', 'tote bag': 'Personal Item', 'stool': 'Personal Item', 'flag': 'Personal Item', 
                        'lollipop': 'Personal Item', 'segway': 'Vehicle', 'arm': 'Body Part', 'writing': 'HAR', 'tight': 'Miscellaneous', 
                        'vespa': 'Vehicle', 'trick': 'Miscellaneous', 'stretch': 'HAR', 'pant': 'Clothing Accessory', 
                        'grocery bag': 'Personal Item', 'floor': 'Miscellaneous', 'car seat': 'Vehicle', 'runner': 'People', 
                        'crate': 'Personal Item', 'helmet': 'Personal Item', 'mower': 'Personal Item', 'skater': 'People', 
                        'gray': 'Color', 'parking meter': 'Urban Infrastructure', 'baseball glove': 'Personal Item', 
                        'jumpsuit': 'Clothing Accessory', 'box': 'Personal Item', 'cardboard box': 'Personal Item', 'package': 'Personal Item'}

    df['Category'] = df['Tag'].map(category_mapping)
    return df

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


def plot_network(graph, word_counts_df):
    # Define category-to-color mapping
    category_to_color = {
        'HAR': 'red', 'Vehicle': 'green', 'People': 'blue', 'Urban Infrastructure': 'black',
        'Color': 'orange', 'Miscellaneous': 'brown', 'Clothing Accessory': 'pink', 
        'Personal Item': 'purple', 'Animal': 'yellow', 'Body Part': 'cyan'
    }

    pos = nx.spring_layout(graph, seed=42)  # Layout for better visualization

    # Create edge traces for the graph with showlegend=False
    edge_trace = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(
            go.Scatter(x=[x0, x1, None], y=[y0, y1, None], mode='lines', 
                       line=dict(width=0.03, color='gray'), hoverinfo='none', showlegend=False)
        )

    # Create the main node trace with showlegend=False
    node_trace = go.Scatter(x=[], y=[], mode='markers', text=[], 
                            marker=dict(size=[], color=[]), opacity=0.7, showlegend=False)
    
    # Add nodes to the trace with corresponding category-based color and size
    for node in graph.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        
        tag_category = word_counts_df.loc[word_counts_df['Tag'] == node, 'Category']
        category = tag_category.values[0] if not tag_category.empty and not pd.isna(tag_category.values[0]) else 'Unknown'
        
        color = category_to_color.get(category, 'gray')
        
        node_trace['marker']['color'] += (color,)
        node_trace['marker']['size'] += (graph.nodes[node].get('size', 1) * 0.03,)
        
        connected_nodes = list(graph.neighbors(node))
        top_connected_nodes_text = "<br>".join(connected_nodes[:10]) if connected_nodes else "None"
        node_trace['text'] += ([f'Node: {node}<br>Category: {category}<br>Connections: {len(connected_nodes)}<br>Top Connected Nodes: {top_connected_nodes_text}'],) 

    # Create legend manually with separate scatter plots representing each category
    legend_traces = []
    for category, color in category_to_color.items():
        legend_traces.append(
            go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color=color), name=category, showlegend=True)
        )
    
    # Create the Plotly figure with the network graph and the category-based legend
    fig = go.Figure(
        data=[*edge_trace, node_trace, *legend_traces], 
        layout=go.Layout(
            title='Network Graph with Legend',
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )

    return fig


def location_based_graph(df, location):
    df_location = df[df['location'] == location]
    co_occurrence_overall = get_co_occurrence(df_location)
    words_count_1, words_cps_1, words_count_2, words_cps_2, words_count_3, words_cps_3 = create_individual_location_data(df)
    words_count_1_location, a, words_count_2_location, b, words_count_3_location, c = create_individual_location_data(df_location)    
    word_counts_df_overall = create_aggregated_data(words_count_1_location, words_count_2_location, words_count_3_location)
    word_counts_df_overall_categorised = category_mapping(word_counts_df_overall)
    words_cps_m, words_cps_f = create_gender_data(df_location)
    graph = create_graph_from_co_occurrence(co_occurrence_overall, word_counts_df_overall_categorised)
    return graph, word_counts_df_overall_categorised, words_cps_1, words_cps_2, words_cps_3, words_cps_m, words_cps_f


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

# HAR - Human Activity Recognition Graph
# for radio button, if you want entire data, pass None as location_id, if you want to specify location, pass location_id=1 or 2 or 3
def create_har_df(df, human_actions, location_id=None):
    if location_id is not None:
        filtered_df = df[df['locationID'] == location_id]
    else:
        filtered_df = df.copy()  # Use the entire DataFrame if no location_id is provided
    
    word_counts_df = get_counts_df(filtered_df)
    har_df = word_counts_df[word_counts_df['Tag'].isin(human_actions)].reset_index(drop=True)
    
    return har_df

def create_har_normalize_counts(df):
    total_counts = df['RAM_Count'].sum()
    df['normalized_count'] = df['RAM_Count'] / total_counts
    return df

def plot_har_bar_graph(df):
    fig = go.Figure([go.Bar(x=df['Tag'], y=df['normalized_count'])])
    fig.update_layout(title='Normalized Occurrences of Human Action Tags', xaxis_title='Tag', yaxis_title='Normalized Occurrences')
    return fig

# Data processing logic for co-occurrence graph
df = pd.read_csv('/Users/hp/Population-Estimation-using-Street-Cam-Footage/YOLO+RAM_merged.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)

# Create a co-occurrence matrix and word counts
co_occurrence_overall = get_co_occurrence(df)
words_count_1, words_cps_1, words_count_2, words_cps_2, words_count_3, words_cps_3 = create_individual_location_data(df)
word_counts_df_overall = create_aggregated_data(words_count_1, words_count_2, words_count_3)
words_cps_m, words_cps_f = create_gender_data(df)
word_counts_df_overall_categorised = category_mapping(word_counts_df_overall)
graph = create_graph_from_co_occurrence(co_occurrence_overall, word_counts_df_overall_categorised)

#graph, word_counts_df_overall, words_cps_1, words_cps_2, words_cps_3, words_cps_m, words_cps_f = location_based_graph(df, 'park')
#graph, word_counts_df_overall, words_cps_1, words_cps_2, words_cps_3, words_cps_m, words_cps_f = location_based_graph(df, 'chase')
#graph, word_counts_df_overall, words_cps_1, words_cps_2, words_cps_3, words_cps_m, words_cps_f = location_based_graph(df, 'dumbo')

# create HAR bar graph
human_actions = ['drive','ride','cross','walk','pick up','stand','carry','catch','jog','spray','push','skate','wash','travel',
                 'clean','wear','crowded','take','run','swab','drag','play','check','stretch']
# Change this to the desired location ID or None for all locations
location_id = 1  
har_df = create_har_df(df, human_actions, location_id)
normalized_har_df = create_har_normalize_counts(har_df)
har_fig = plot_har_bar_graph(normalized_har_df)

#Create Dash app
app = Dash(__name__)

# Layout with network graph and two bar charts
app.layout = html.Div([
    html.H1("Interactive Network Graph with Multiple Hover-based Bar Charts"),
    html.Div([
        dcc.Graph(
            id='network-graph',
            figure=plot_network(graph, word_counts_df_overall_categorised),
            hoverData={'points': [{'text': 'example_tag'}]}  # Default hover data
        )
    ], style={'width': '70%', 'display': 'inline-block'}),

    # placeholder for the HAR bar chart
    html.Div([
        dcc.Graph(
            id='har_bar_graph',
            figure=plot_har_bar_graph(normalized_har_df)
        )
    ], style={'width': '70%', 'display': 'inline-block'}),

    html.Div([
        dcc.Graph(id='bar-graph-1'),  # Placeholder for the first bar chart
        dcc.Graph(id='bar-graph-2')  # Placeholder for the second bar chart
    ], style={'width': '30%', 'display': 'inline-block'})
])

# Callback to update both bar charts based on hover data from the network graph
@callback(
    [Output('bar-graph-1', 'figure'), Output('bar-graph-2', 'figure')],
    [Input('network-graph', 'hoverData')]
)
def update_bar_charts(hover_data):
    tag = hover_data['points'][0]['text'].split("<br>")[0].replace("Node: ", "").strip("['") if hover_data and 'points' in hover_data else None
    
    if tag:
        # Get data for the first bar chart
        tag_data_location = get_tag_data_without_label(tag, words_cps_1, words_cps_2, words_cps_3)
        
        # Generate the first bar chart
        bar_fig_1 = go.Figure(
            data=[
                go.Bar(x=list(tag_data_location.keys()), y=list(tag_data_location.values()), marker=dict(color=['red', 'green', 'black']))
            ],
            layout=go.Layout(
                title=f'RAM Count for Tag: {tag}',
                xaxis=dict(title='Location'),
                yaxis=dict(title='Count per Second'),
                showlegend=False
            )
        )

        # Generate the second bar chart (assuming you have another data source)
        tag_data_gender = get_tag_data_without_label(tag, words_cps_m, words_cps_f)
        
        bar_fig_2 = go.Figure(
            data=[
                go.Bar(x=list(tag_data_gender.keys()), y=list(tag_data_gender.values()), marker=dict(color=['blue', 'pink']))
            ],
            layout=go.Layout(
                title=f'RAM Count for Tag: {tag}',
                xaxis=dict(title='Gender'),
                yaxis=dict(title='Count per Second'),
                showlegend=False
            )
        )

        return [bar_fig_1, bar_fig_2]
    else:
        return [go.Figure(), go.Figure()]  # Empty figures if no tag is hovered over

if __name__ == '__main__':
    app.run(debug=True)