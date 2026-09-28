import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
import joblib
import gc

chunk_size = 1_000_000
total_rows_to_read = 30_000_000
print(f"1. Reading exactly {total_rows_to_read} in {chunk_size}-row chunks...")
filtered_chunks = []

# Pandas natively stops reading after 'nrows' is hit
for chunk in pd.read_csv('data/ratings.csv', chunksize=chunk_size, nrows=total_rows_to_read, usecols=['userID', 'animeID', 'rating']):
    # Filter out 0/unrated scores instantly
    chunk = chunk[chunk['rating'] >= 1.0]
    filtered_chunks.append(chunk)

ratings_df = pd.concat(filtered_chunks, ignore_index=True)

del filtered_chunks
gc.collect()
print(f"Chunking complete. Valid rows loaded: {len(ratings_df):,}")

print("2. Applying proportional quality filters...")
# Lowered thresholds specifically for the 2M row subset
anime_counts = ratings_df['animeID'].value_counts()
popular_anime = anime_counts[anime_counts >= 50].index  # Was 1000

user_counts = ratings_df['userID'].value_counts()
active_users = user_counts[user_counts >= 20].index     # Was 200

ratings_df = ratings_df[
    ratings_df['animeID'].isin(popular_anime) & 
    ratings_df['userID'].isin(active_users)
]
print(f"Data filtered. Remaining rows for matrix: {len(ratings_df):,}")

print("3. Building Sparse Matrix directly...")
anime_category = ratings_df['animeID'].astype('category')
user_category = ratings_df['userID'].astype('category')

row_indices = anime_category.cat.codes.values
col_indices = user_category.cat.codes.values
rating_values = ratings_df['rating'].values

anime_matrix_sparse = coo_matrix((rating_values, (row_indices, col_indices))).tocsr()
anime_index_list = list(anime_category.cat.categories)

del ratings_df, anime_category, user_category, row_indices, col_indices, rating_values
gc.collect()

print("4. Training KNN model...")
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
model_knn.fit(anime_matrix_sparse)

print("5. Saving model and indices to disk...")
joblib.dump(model_knn, 'knn_model.pkl')
joblib.dump(anime_index_list, 'anime_index.pkl')

print("Success! Brain is saved.")