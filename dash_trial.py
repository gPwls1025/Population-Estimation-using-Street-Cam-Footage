import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objects as go
from process import update_co_occurrence, create_filtered_word_counts_df, get_word_counts_and_co_occurrence
from plots import create_graph_from_co_occurrence, plot_network, get_tag_data_without_label, plot_tag_counts_by_location

# Load your data
file_path = 'data/'
df = pd.read_csv(file_path + 'YOLO+RAM_merged.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)

word_counts_df_filtered, co_occurrence_overall, words_cps_1, words_cps_2, words_cps_3 = get_word_counts_and_co_occurrence(df)

words_cps_1.name = 'park'
words_cps_2.name = 'chase'
words_cps_3.name = 'dumbo'

# Initialize the Dash app
app = dash.Dash(__name__)

location_labels = {'All Locations': 'All Locations', 'Park': 1, 'Chase': 2, 'Dumbo': 3}

# Define the layout of the app
app.layout = html.Div([
    html.H1('VizML Dashboard', style={'text-align': 'center'}),
    dcc.Dropdown(
        id='location-dropdown',
        options=[{'label': loc, 'value': value} for loc, value in location_labels.items()],
        value='All',
        style={'width': '50%'}
    ),
    html.Div([
        dcc.Graph(id='network-graph', style={'width': '60%', 'display': 'inline-block'}),
        dcc.Graph(id='bar-chart', style={'width': '40%', 'display': 'inline-block'})
    ])
])

# Callback to update the network graph based on location selection
@app.callback(
    Output('network-graph', 'figure'),
    [Input('location-dropdown', 'value')]
)
def update_network_graph(selected_location):
    # if selected_location == 'All Locations':
    filtered_data = word_counts_df_filtered
    # else:
    #     filtered_data = word_counts_df_filtered[word_counts_df_filtered['location'] == selected_location]
    
    graph = create_graph_from_co_occurrence(co_occurrence_overall, filtered_data)
    return plot_network(graph, "Network Graph for " + selected_location)

# Callback to update the bar chart based on node hover in the network graph
@app.callback(
    Output('bar-chart', 'figure'),
    [Input('network-graph', 'hoverData'), Input('location-dropdown', 'value')]
)
def update_bar_chart(hover_data, selected_location):
    if hover_data and 'points' in hover_data:
        node_tag = hover_data['points'][0]['text'].split("<br>")[0].replace("Node: ", "").strip()
        tag_data = word_counts_df_filtered[word_counts_df_filtered['Tag'] == node_tag]

        # if selected_location != 'All':
        #     tag_data = tag_data[tag_data['location'] == selected_location]
        
        fig = go.Figure(
            data=[
                go.Bar(x=tag_data['Tag'], y=tag_data['RAM_Count'], marker=dict(color='blue'))
            ],
            layout=go.Layout(
                title=f'RAM Count for Tag: {node_tag}',
                xaxis=dict(title='Tag'),
                yaxis=dict(title='RAM Count'),
                showlegend=False
            )
        )
        return fig

    return go.Figure()  # Return an empty figure if no node is hovered over

if __name__ == '__main__':
    app.run_server(debug=True)
