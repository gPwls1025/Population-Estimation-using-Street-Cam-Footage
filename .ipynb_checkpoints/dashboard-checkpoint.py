
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
        return pd.read_csv(f"/data/{path}", index_col=0)
    
    # Function to get top N co-occurring pairs
    def top_n_cooccur_pairs(df, top_n):
        pairs = df.stack()
        pairs = pairs[pairs.index.get_level_values(0) < pairs.index.get_level_values(1)]
        top_pairs = pairs.sort_values(ascending=False).head(top_n)
        return top_pairs
    
    # Displaying the top N co-occurring pairs
    if st.button('Show Top Pairs'):
        df = load_data(location) 
        top_pairs = top_n_cooccur_pairs(df, top_n)
        top_pairs.index = [" & ".join(map(str, idx)) for idx in top_pairs.index]
    
        fig = px.bar(top_pairs, x=top_pairs.values, y=top_pairs.index, orientation='h',
                     labels={'y': 'Pairs', 'x': 'Counts'},
                     title=f"Top {top_n} Co-occurring Pairs in {location}")
        
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()