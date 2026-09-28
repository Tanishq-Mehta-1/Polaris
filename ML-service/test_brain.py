import joblib
import pandas as pd

print("Waking up the brain...")
model_knn = joblib.load('knn_model.pkl')
anime_index_list = joblib.load('anime_index.pkl')
anime_df = pd.read_csv('data/animes.csv')

# The KNN model stores the training matrix internally in the _fit_X attribute
saved_matrix = model_knn._fit_X 

def get_recommendations(target_anime_id, num_recommendations=5):
    if target_anime_id not in anime_index_list:
        return print(f"Error: Anime ID {target_anime_id} is not in our row subset.")

    # 1. Find the row number
    query_index = anime_index_list.index(target_anime_id)

    # 2. Grab the anime's vector from the saved matrix and calculate distances
    distances, indices = model_knn.kneighbors(
        saved_matrix[query_index], 
        n_neighbors=num_recommendations + 1
    )

    # 3. Print the results
    target_title = anime_df.loc[anime_df['animeID'] == target_anime_id, 'title'].values[0]
    print(f"\nBecause you liked '{target_title}':")

    for i in range(1, len(distances.flatten())):
        rec_id = anime_index_list[indices.flatten()[i]]
        rec_title = anime_df.loc[anime_df['animeID'] == rec_id, 'title'].values[0]
        similarity = (1 - distances.flatten()[i]) * 100
        print(f"{i}. {rec_title} (Similarity: {similarity:.1f}%)")

# Test with a popular ID, e.g., 1535 (Death Note) or 16498 (Attack on Titan)
get_recommendations(1)