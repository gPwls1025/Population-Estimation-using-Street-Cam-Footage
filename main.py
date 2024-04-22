import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from process import process_data
from plots import create_graph_from_co_occurrence, plot_network, plot_tag_counts_by_location
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
    location_labels = {'All': 'All Locations', 'park': 'Park', 'chase': 'Chase', 'dumbo': 'Dumbo'}
    # Add location labels in the selectbox
    location = st.sidebar.selectbox("Pick your location", location_labels.keys(), format_func=lambda x: location_labels[x])

    # filter data based on location - group by location 
    if location == 'All':
        df_filtered = df
    elif location == 'park':
        df_filtered = df[df['location'] == 'park']
    elif location == 'chase':
        df_filtered = df[df['location'] == 'chase']
    elif location == 'dumbo':
        df_filtered = df[df['location'] == 'dumbo']

    # create columns for graph insertion 
    col1, col2 = st.columns([1.5,1])

    # plot graphs
    with col1:
        st.subheader("Network Grpah of Street Cam Footage")
        graph_overall = create_graph_from_co_occurrence(co_occurrence_overall, word_counts_df_filtered)
        st.plotly_chart(plot_network(graph_overall, "Network Graph of Street Cam Footage"), use_container_width=True, height=500)

    with col2:
        st.subheader("Bar char of human actions detected")
        if location == 'All': 
            fig = px.bar(word_counts_df_filtered, x='Tag', y='RAM_Count', title='# of Tags by Location')
            st.plotly_chart(fig, use_container_width=True, height=200)
        else: 
            fig = plot_tag_counts_by_location(df_filtered, location)
            st.plotly_chart(fig, use_container_width=True)
        

if __name__ == "__main__":
    main()

