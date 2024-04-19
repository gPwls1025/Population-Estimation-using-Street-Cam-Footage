
import streamlit as st
import pandas as pd
import plotly.express as px

def main(): 
    
    st.title('Top N Co-occurring Pairs Dashboard')
    location = st.selectbox('Select Location', ['Chase', 'Dumbo', 'Park'])
    
    # Slider to select top N
    top_n = st.slider('Select Top N', min_value=1, max_value=50, value=5)
    
    def load_data(loc):
        path = f"{loc.lower()}_cooccurrence.csv"
        return pd.read_csv(f"./data/{path}", index_col=0)
    
    # Function to get top N co-occurring pairs
    def top_n_cooccur_pairs(df, top_n):
    
        pairs = df.stack().reset_index()
        pairs.columns = ['Term1', 'Term2', 'Co-occurrence']
        pairs = pairs[pairs['Term1'] < pairs['Term2']]
        top_pairs = pairs.sort_values(by='Co-occurrence', ascending=False).head(top_n)
        
        return top_pairs
    
    # Displaying the top N co-occurring pairs
    if st.button('Show Top Pairs'):
        df = load_data(location) 
        top_pairs = top_n_cooccur_pairs(df, top_n)
        top_pairs['Pair'] = top_pairs['Term1'] + " & " + top_pairs['Term2']
        
        # Calculate figure height: base height + (number of bars * height per bar)
        figure_height = max(400, 20 * len(top_pairs))
        fig = px.bar(top_pairs, x='Co-occurrence', y='Pair', orientation='h',
                     labels={'Pair': 'Pairs', 'Co-occurrence': 'Counts'},
                     title=f"Top {top_n} Co-occurring Pairs in {location}")
        
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=figure_height)
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()