import pandas as pd
import plotly.graph_objects as go
import networkx as nx
import json
import dash_bootstrap_components as dbc
import plotly.express as px
from dash_bootstrap_templates import load_figure_template
from dash import Dash, html, dcc, callback, Input, Output, State
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
    df = df.copy()
    df.loc[:, 'man'] = df['RAM_Tags'].str.contains(r'\bman\b|\bboy\b', case=False, regex=True).astype(int)
    df.loc[:, 'woman'] = df['RAM_Tags'].str.contains(r'\bwoman\b|\bgirl\b', case=False, regex=True).astype(int)
    man_df = df[(df['man'] == 1) & (df['woman'] == 0)]
    woman_df = df[(df['man'] == 0) & (df['woman'] == 1)]
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


def plot_network(graph, word_counts_df, height=650):
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
            template=theme,
            title='Network Graph with Legend',
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=height
        )
    )
    return fig


def create_network_and_comatrix(df, location=None):
    df_location = df[df['location'] == location] if location else df
    co_occurrence_overall = get_co_occurrence(df_location)
    words_count_1_location, words_cps_1, words_count_2_location, words_cps_2, words_count_3_location, words_cps_3 = create_individual_location_data(df_location)
    word_counts_df_overall = create_aggregated_data(words_count_1_location, words_count_2_location, words_count_3_location)
    word_counts_df_overall_categorized = category_mapping(word_counts_df_overall)
    words_cps_m, words_cps_f = create_gender_data(df_location)
    graph = create_graph_from_co_occurrence(co_occurrence_overall, word_counts_df_overall_categorized)
    
    return graph, word_counts_df_overall_categorized, words_cps_1, words_cps_2, words_cps_3, words_cps_m, words_cps_f

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
def create_har_df(df, human_actions, location=None):
    if location is not None:
        filtered_df = df[df['location'] == location]
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
    fig.update_layout(title='Normalized Occurrences of Human Action Tags', xaxis_title='Tag', yaxis_title='Normalized Occurrences', template=theme)
    return fig

def process_interaction():
    chase_df = pd.read_csv('/Users/hp/Population-Estimation-using-Street-Cam-Footage/chase_cooccurrence.csv', index_col=0)
    park_df = pd.read_csv('/Users/hp/Population-Estimation-using-Street-Cam-Footage/park_cooccurrence.csv', index_col=0)
    dumbo_df = pd.read_csv('/Users/hp/Population-Estimation-using-Street-Cam-Footage/dumbo_cooccurrence.csv', index_col=0)

    people = [
        'baby',
        'boy',
        'businessman',
        'child',
        'construction worker',
        'couple',
        'daughter',
        'girl',
        'man',
        'mother',
        'nun',
        'officer',
        'pedestrian',
        'person',
        'protester',
        'runner',
        'skater',
        'skateboarder',
        'student',
        'woman'
    ]

    chase_car_people_interaction = chase_df.loc['car', people].sum()
    park_car_people_interaction = park_df.loc['car', people].sum()
    dumbo_car_people_interaction = dumbo_df.loc['car', people].sum()

    total_interaction = chase_car_people_interaction + park_car_people_interaction + dumbo_car_people_interaction

    chase_interaction_val =  chase_car_people_interaction/ total_interaction
    park_interaction_val = park_car_people_interaction / total_interaction
    dumbo_interaction_val = dumbo_car_people_interaction / total_interaction
    
    return chase_interaction_val, park_interaction_val, dumbo_interaction_val
    
def plot_interaction_pie(chase, park, dumbo):
    data = {
        'Categories': ['chase', 'park', 'dumbo'],
        'Values': [chase, park, dumbo]
    }

    df_data = pd.DataFrame(data)

    fig = px.pie(
        df_data,
        names='Categories', 
        values='Values', 
        title='People to Vehicle Interaction',
        color_discrete_sequence=px.colors.qualitative.Plotly,
        
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig


# I am processing all the required dataframes here
def process_dfs(df):
    # create HAR bar graph
    human_actions = ['drive','ride','cross','walk','pick up','stand','carry','catch','jog','spray','push','skate','wash','travel',
                    'clean','wear','crowded','take','run','swab','drag','play','check','stretch']

    # Precompute graphs and data for each location
    network_dfs = {}
    har_dfs = {}
    locations = df['location'].unique()

    for loc in locations:
        graph, word_counts_df_filtered_categorized, words_cps_1, words_cps_2, words_cps_3, words_cps_m, words_cps_f = create_network_and_comatrix(df, loc)
        network_dfs[loc] = (graph, word_counts_df_filtered_categorized, words_cps_1, words_cps_2, words_cps_3, words_cps_m, words_cps_f)

        har_df = create_har_df(df, human_actions, loc)
        normalized_har_df = create_har_normalize_counts(har_df)
        har_dfs[loc] = normalized_har_df
    
    # Also do it for all location
    graph, word_counts_df_filtered_categorized, words_cps_1, words_cps_2, words_cps_3, words_cps_m, words_cps_f = create_network_and_comatrix(df)
    network_dfs['all'] = (graph, word_counts_df_filtered_categorized, words_cps_1, words_cps_2, words_cps_3, words_cps_m, words_cps_f)

    har_df = create_har_df(df, human_actions)
    normalized_har_df = create_har_normalize_counts(har_df)
    har_dfs['all'] = normalized_har_df
    
    chase_interaction, park_interaction, dumbo_interaction = process_interaction()
    return network_dfs, har_dfs, chase_interaction, park_interaction, dumbo_interaction


# Data processing logic for co-occurrence graph
df = pd.read_csv('/Users/hp/Population-Estimation-using-Street-Cam-Footage/YOLO+RAM_merged.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)

network_dfs, har_dfs, chase_interaction, park_interaction, dumbo_interaction = process_dfs(df)

# Let's create template 
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
theme = "BOOTSTRAP"
load_figure_template(theme)

title = dcc.Markdown(
    """
### Color - Background
------------
"""
)

color_bg = html.Div(
    [
        html.P("bg-primary", className="bg-primary"),
        html.P("bg-secondary", className="bg-secondary"),
        html.P("bg-success", className="bg-success"),
        html.P("bg-danger", className="bg-danger"),
        html.P("bg-warning", className="bg-warning"),
        html.P("bg-info", className="bg-info"),
        html.P("bg-light", className="bg-light"),
        html.P("bg-dark", className="bg-dark"),
        html.P("bg-transparent", className="bg-transparent"),
    ]
)

color_bg_gradient = html.Div(
    [
        html.P("bg-primary text-white py-4", className="bg-primary text-white py-4"),
        html.P(
            "bg-primary  bg-gradient text-white py-4",
            className="bg-primary bg-gradient text-white py-4",
        ),
    ]
)


border_direction = dbc.Card(
    [
        dbc.CardHeader("Border Direction"),
        html.Div(
            [
                html.P("border ", className="border "),
                html.P("border-top ", className="border-top "),
                html.P("border-end ", className="border-end "),
                html.P("border-bottom ", className="border-bottom "),
                html.P("border-start ", className="border-start "),
            ],
            className="p-4",
        ),
    ],
    className="my-4",
)


border_direction_0 = dbc.Card(
    [
        dbc.CardHeader("Border Direction 0"),
        html.Div(
            [
                html.P("border border-0", className="border border-0 "),
                html.P("border border-top-0 ", className="border border-top-0 "),
                html.P("border border-end-0 ", className="border border-end-0 "),
                html.P("border border-bottom-0 ", className="border border-bottom-0 "),
                html.P("border border-start-0 ", className="border border-start-0 "),
            ],
            className="p-4",
        ),
    ],
    className="my-4",
)


border_color = dbc.Card(
    [
        dbc.CardHeader("Border Color"),
        html.Div(
            [
                html.P("border border-primary ", className="border border-primary "),
                html.P(
                    "border border-secondary ", className="border border-secondary "
                ),
                html.P("border border-success ", className="border border-success "),
                html.P("border border-danger ", className="border border-danter "),
                html.P("border border-warning ", className="border border-warning "),
                html.P("border border-info ", className="border border-info "),
                html.P("border border-light", className="border border-light "),
                html.P("border border-dark  ", className="border border-dark "),
                html.P("border border-white  ", className="border border-white "),
            ],
            className="p-4",
        ),
    ],
    className="my-4",
)

border_size = dbc.Card(
    [
        dbc.CardHeader("Border Size"),
        html.Div(
            [
                html.P("border border-1 ", className="border border-1 "),
                html.P("border border-2 ", className="border border-2 "),
                html.P("border border-3 ", className="border border-3 "),
                html.P("border border-4 ", className="border border-4 "),
                html.P("border border-5 ", className="border border-5 "),
            ],
            className="p-4",
        ),
    ],
    className="my-4",
)

# Layout with network graph and two bar charts
app.layout = html.Div([
    # Title
    html.Div(
        html.H1("Population Estimation using Street Camera Footage"),
        className="bg-secondary bg-gradient text-center text-white py-4"
        # style={'padding': '20px', 'margin-left': '160px', 'margin-top': '20px'} 
    ),
    # Dropdown
    html.Div([
        dcc.Dropdown(
            id='location-dropdown',
            options=[
                {'label': 'Park', 'value': 'park'},
                {'label': 'Chase', 'value': 'chase'},
                {'label': 'Dumbo', 'value': 'dumbo'},
                {'label': 'All Locations', 'value': 'all'}
            ],
            value='all',  # Default value to show all locations
            placeholder="Select a location",
            clearable=False,
            style={'width': '50%', 'display': 'inline-block', 'margin-top': '5px'} # 'padding':'20px'
        )
    ]),
    # Network
    html.Div([
        dcc.Graph(
            id='network-graph',
            figure=plot_network(network_dfs['all'][0], network_dfs['all'][1]),
            hoverData={'points': [{'text': 'example_tag'}]}  # Default hover data
        )
    ], 
        style={'width': '70%', 'display': 'inline-block'},
        className="border border-light border-4"),
    # Interactive bars
    html.Div([
            dcc.Graph(id='bar-graph-1', style={'width': '100%', 'display': 'inline-block', 'height': '300px'}),
            dcc.Graph(id='bar-graph-2', style={'width': '100%', 'display': 'inline-block', 'height': '300px'})
        ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'},
        className="border border-light border-4"),
    # HAR
    html.Div([
        dcc.Graph(
            id='har-graph',
            figure=plot_har_bar_graph(har_dfs['all'])
        )
    ], style={'width': '70%', 'display': 'inline-block'},
    className="border border-light border-4"),
    # People-Car Interaction with initial pie chart
    html.Div(
        id='interaction-display', 
        children=dcc.Graph(id='interaction-pie-chart', figure=plot_interaction_pie(chase_interaction, park_interaction, dumbo_interaction)),
        style={'width': '30%', 'display': 'inline-block'},
        className="border border-light border-4"
    ),

])

##########################
### Plot Network graph ###
##########################
@app.callback(
    Output('network-graph', 'figure'),
    [Input('location-dropdown', 'value')]
)
def update_network_graph(selected_location):
    
    graph, word_counts_df_filtered_categorized, words_cps_1_new, words_cps_2_new, words_cps_3_new, words_cps_m_new, words_cps_f_new = network_dfs[selected_location]
    
    return plot_network(graph, word_counts_df_filtered_categorized)

# Callback for the conditional People-Car Interaction display
@app.callback(
    Output('interaction-display', 'children'),
    [Input('location-dropdown', 'value')]
)
def update_interaction_display(selected_location):
    if selected_location == 'all':
        # Return the pie chart
        pie_chart_figure = plot_interaction_pie(chase_interaction, park_interaction, dumbo_interaction)
        return dcc.Graph(id='interaction-pie-chart', figure=pie_chart_figure)
    else:
        # Return text
        interaction_val = {
            'park': park_interaction,
            'chase': chase_interaction,
            'dumbo': dumbo_interaction
        }[selected_location]
        return html.Div(
            f'Car to People Interaction for {selected_location.capitalize()}: {interaction_val:.2%}',
            style={
                'fontSize': '24px',  
                'textAlign': 'center',  
                'margin': '10px 0',  
                'position': 'relative',
                'top': '-200px' 
            }
        )


##########################
### Plot HAR bar graph ###
##########################
@app.callback(
    Output('har-graph', 'figure'),
    [Input('location-dropdown', 'value')]
)
def update_additional_graph(selected_location):
    # Example: Suppose you want to create a simple histogram of the 'RAM_Count'
    if selected_location == 'all':
        df_to_plot = har_dfs['all']
    else:
        df_to_plot = har_dfs[selected_location]

    return plot_har_bar_graph(df_to_plot)

###############################################
### Interactive bar charts on network graph ###
###############################################
# Callback to update bar charts based on hover data from network graph
@app.callback(
    [Output('bar-graph-1', 'figure'), Output('bar-graph-2', 'figure')],
    [Input('network-graph', 'hoverData')],
    [State('location-dropdown', 'value')]
)
def update_bar_charts(hover_data, selected_location):
    if hover_data:
        tag = hover_data['points'][0]['text'].split("<br>")[0].replace("Node: ", "").strip("['") if 'points' in hover_data and hover_data['points'] else None
        if tag:
            # Fetch the necessary data using the currently selected location
            network_data = network_dfs[selected_location] if selected_location in network_dfs else network_dfs['all']
            tag_data_gender = get_tag_data_without_label(tag, network_data[5], network_data[6])
            
            network_data = network_dfs['all']
            tag_data_location = get_tag_data_without_label(tag, network_data[2], network_data[3], network_data[4])

            bar_fig_1 = go.Figure(data=[
                go.Bar(x=list(tag_data_location.keys()), y=list(tag_data_location.values()), marker=dict(color=['red', 'green', 'black']))
            ])

            bar_fig_1.update_layout(
                title={'text': 'Location Distribution', 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                xaxis_title = 'Location',
                yaxis_title = 'Counts'
            )
            
            bar_fig_2 = go.Figure(data=[
                go.Bar(x=list(tag_data_gender.keys()), y=list(tag_data_gender.values()), marker=dict(color=['blue', 'pink']))
            ])

            bar_fig_2.update_layout(
                title={'text': 'Gender Distribution', 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                xaxis_title = 'Gender',
                yaxis_title = 'Counts'
            )
            
            return [bar_fig_1, bar_fig_2]
    return [go.Figure(), go.Figure()]  # Return empty figures if no hover data or tag not found

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)