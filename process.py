import pandas as pd

def update_co_occurrence(df, co_occurrence_matrix):
    for _, row in df.iterrows():
        tags = row['RAM_Tags'].split(' | ')
        for i in range(len(tags)):
            for j in range(i+1, len(tags)):
                co_occurrence_matrix.at[tags[i], tags[j]] += 1
                co_occurrence_matrix.at[tags[j], tags[i]] += 1


def create_filtered_word_counts_df(df, location_id):
    filtered_df = df[df['locationID'] == location_id]
    text = ' | '.join(filtered_df['RAM_Tags'])
    words = text.split('|')
    words = [word.strip().lower() for word in words]
    word_counts = pd.Series(words).value_counts()
    
    word_counts_df = pd.DataFrame(word_counts)
    word_counts_df.reset_index(inplace=True)
    word_counts_df.columns = ['Tag', 'RAM_Count']
    word_counts_df_filtered = word_counts_df[~((word_counts_df['RAM_Count'] > 0.75 * len(filtered_df)) | (word_counts_df['RAM_Count'] <= 4))]
    
    # Calculate normalized counts
    normalized_counts = pd.DataFrame({'Tag': word_counts_df_filtered['Tag'], 'RAM_Count': word_counts_df_filtered['RAM_Count'] / len(filtered_df)})
    
    return word_counts_df_filtered, normalized_counts


def get_word_counts_and_co_occurrence(df):
    text = ' | '.join(df['RAM_Tags'])
    words = text.split('|')
    words = [word.strip().lower() for word in words]

    # Count the occurrences of each word
    word_counts = pd.Series(words).value_counts()

    #Make a df for storage
    word_counts_df = pd.DataFrame(word_counts)
    word_counts_df.reset_index(inplace=True)
    word_counts_df.columns = ['Tag','RAM_Count']

    all_tags = set()
    for tags in df['RAM_Tags']:
        all_tags.update(tags.split(' | '))
    all_tags = sorted(list(all_tags))

    # Creating co-occurrence matrices
    co_occurrence_overall = pd.DataFrame(index=all_tags, columns=all_tags).fillna(0)

    # Updating co-occurrence matrices
    update_co_occurrence(df, co_occurrence_overall) #This makes the co-occurence matrix

    #word_counts_df_filtered = word_counts_df[~((word_counts_df['RAM_Count']>0.75*len(df)) | (word_counts_df['RAM_Count'] <= 4))]

    # Assuming df is your DataFrame
    # Call the function for each locationID
    words_count_1, words_cps_1 = create_filtered_word_counts_df(df, 1)
    words_count_2, words_cps_2 = create_filtered_word_counts_df(df, 2)
    words_count_3, words_cps_3 = create_filtered_word_counts_df(df, 3)

    words_count_1['RAM_Count'] *= (1/3)
    words_count_2['RAM_Count'] *= (1/3)
    words_count_3['RAM_Count'] *= (1/3)

    merged_df_location = pd.concat([words_count_1, words_count_2, words_count_3], ignore_index=True)
    grouped_df_location = merged_df_location.groupby('Tag')['RAM_Count'].sum().reset_index()

    return grouped_df_location, co_occurrence_overall, words_cps_1, words_cps_2, words_cps_3

