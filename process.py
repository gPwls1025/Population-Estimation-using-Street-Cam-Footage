import pandas as pd

def process_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path, encoding="ISO-8859-1")
    
    # Drop the 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    
    # Concatenate 'RAM_Tags' into a single string
    text = ' | '.join(df['RAM_Tags'])
    
    # Split the text into words and preprocess
    words = [word.strip().lower() for word in text.split('|')]
    
    # Count the occurrences of each word
    word_counts = pd.Series(words).value_counts()
    
    # Create a DataFrame for word counts
    word_counts_df = pd.DataFrame(word_counts).reset_index()
    word_counts_df.columns = ['Tag', 'RAM_Count']
    
    # Exclude words that occur too frequently
    threshold = 0.75 * len(df)
    word_counts_filtered = word_counts[word_counts <= threshold]
    
    # Filter the word counts DataFrame
    word_counts_df_filtered = word_counts_df[~((word_counts_df['RAM_Count'] > 0.75 * len(df)) | (word_counts_df['RAM_Count'] <= 4))]
    
    # Define human actions
    human_actions = ['drive','ride','cross','walk','pick up','stand','carry','catch','jog','spray','push','skate','wash','travel',
                     'clean','wear','crowded','take','run','swab','drag','play','check','stretch']
    
    # Create columns for human actions
    for action in human_actions:
        df[action] = df['RAM_Tags'].str.contains(r'\b' + action + r'\b', case=False, regex=True).astype(int)
    
    # Create co-occurrence matrices
    all_tags = set()
    for tags in df['RAM_Tags']:
        all_tags.update(tags.split(' | '))
    all_tags = sorted(list(all_tags))

    # Creating co-occurrence matrices
    co_occurrence_overall = pd.DataFrame(index=all_tags, columns=all_tags).fillna(0)

    # Function to update co-occurrence matrices
    def update_co_occurrence(df, co_occurrence_matrix):
        for _, row in df.iterrows():
            tags = row['RAM_Tags'].split(' | ')
            for i in range(len(tags)):
                for j in range(i+1, len(tags)):
                    co_occurrence_matrix.at[tags[i], tags[j]] += 1
                    co_occurrence_matrix.at[tags[j], tags[i]] += 1

    # Updating co-occurrence matrices
    update_co_occurrence(df, co_occurrence_overall)
    
    return df, word_counts_df_filtered, co_occurrence_overall
