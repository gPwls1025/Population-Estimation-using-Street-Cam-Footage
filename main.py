import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from process import process_data
from plots import create_graph_from_co_occurrence, plot_network
import warnings
warnings.filterwarnings("ignore")

def main(): 
    # set page title
    st.set_page_config(page_title='VizML Dashboard', layout='wide')
    st.title('VizML Dashboard')
    # move title up a bit
    st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

    # upload file
    file_path = '/Users/hp/Population-Estimation-using-Street-Cam-Footage/'
    df, word_counts_df_filtered, co_occurrence_overall = process_data(file_path + 'YOLO+RAM_merged.csv')

    # Display the sidebar
    st.sidebar.header("Choose the filters")
    location = st.sidebar.multiselect("Pick your location", df['location'].unique())

    # filter data based on location
    if location == ['park']:
        df_filtered = df[df['location'] == 'park']
    elif location == ['chase']:
        df_filtered = df[df['location'] == 'chase']
    else:
        df_filtered = df[df['location'] == 'dumbo']

    # Creating graphs from co-occurrence matrices
    graph_overall = create_graph_from_co_occurrence(co_occurrence_overall, word_counts_df_filtered)

    # plot network graph
    plot_network(graph_overall, "Network Graph")

if __name__ == "__main__":
    main()

