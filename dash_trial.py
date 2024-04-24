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
    dcc.Location(id='url', refresh=False),
    html.Div([
        dcc.Dropdown(
            id='location-dropdown',
            options=[{'label': label, 'value': value} for label, value in location_labels.items()],
            value='All',
            style={'width': '100%'}
        )
    ], style={'padding': 10}),

    html.Div(id='graphs-container')
])

# Callback to update graphs based on selected location
@app.callback(
    Output('graphs-container', 'children'),
    [Input('location-dropdown', 'value')]
)
def update_graphs(selected_location):
    if selected_location == 'All':
        df_filtered = df
    else:
        df_filtered = df[df['location'] == selected_location.lower()]

    # Network graph
    graph_overall = create_graph_from_co_occurrence(co_occurrence_overall, word_counts_df_filtered)
    network_fig = plot_network(graph_overall, "Network Graph of Street Cam Footage")

    # Bar chart
    if selected_location == 'All':
        fig = px.bar(word_counts_df_filtered, x='Tag', y='RAM_Count', title='# of Tags by Location')
    else:
        fig = plot_tag_counts_by_location(df_filtered, selected_location)

    return [
        html.Div([
            dcc.Graph(id='network-graph', figure=network_fig, style={'height': '500px', 'width': '100%'})
        ], style={'display': 'inline-block', 'width': '60%'}),
        html.Div([
            dcc.Graph(id='tag-counts-graph', figure=fig, style={'height': '200px', 'width': '100%'})
        ], style={'display': 'inline-block', 'width': '40%'})
    ]

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
