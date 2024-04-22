import streamlit as st
import pandas as pd
import plotly.express as px

def main(): 
    st.title('Top N Co-occurring Pairs Dashboard')
    location = st.selectbox('Select Location', ['Chase', 'Dumbo', 'Park'])
    
    # Slider to select top N
    top_n = st.slider('Select Top N', min_value=1, max_value=50, value=5)

    person_associated_words = [
        'baby', 'boy', 'businessman', 'child', 'construction worker', 'couple', 'daughter', 
        'girl', 'man', 'mother', 'nun', 'officer', 'pedestrian', 'person', 'protester', 
        'runner', 'skater', 'skateboarder', 'student', 'woman'
    ]

    def load_data(loc):
        path = f"{loc.lower()}_cooccurrence.csv"
        return pd.read_csv(f"./data/{path}", index_col=0)

    def top_n_cooccur_pairs(df, top_n):
        pairs = df.stack().reset_index()
        pairs.columns = ['Term1', 'Term2', 'Co-occurrence']
        pairs = pairs[pairs['Term1'] < pairs['Term2']]
        top_pairs = pairs.sort_values(by='Co-occurrence', ascending=False).head(top_n)
        return top_pairs

    # New section for total co-occurrence with a specific term and associated persons
    st.subheader('Total Co-occurrences with People')
    term = st.text_input('Enter a term (e.g., car)')
    
    # Displaying the top N co-occurring pairs
    if st.button('Show Results'):
        df = load_data(location) 
        top_pairs = top_n_cooccur_pairs(df, top_n)
        top_pairs['Pair'] = top_pairs['Term1'] + " & " + top_pairs['Term2']
        
        figure_height = max(400, 20 * len(top_pairs))
        fig = px.bar(top_pairs, x='Co-occurrence', y='Pair', orientation='h',
                     labels={'Pair': 'Pairs', 'Co-occurrence': 'Counts'},
                     title=f"Top {top_n} Co-occurring Pairs in {location}")
        
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=figure_height)
        st.plotly_chart(fig)

        if term:  # Make sure there's a term to search for
            df = load_data(location)
            total_interactions = df.loc[term, person_associated_words].sum()
            st.write(f'Total interactions for {term} and people:', total_interactions)
        else:
            st.write('Please enter a term to calculate interactions.')

if __name__ == "__main__":
    main()
