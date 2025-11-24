import os
import pandas as pd

# df_core is the dataset I am going to use to create the graph and do PageRank
# The graph will have way less noise, it is going to be more connected and interesting
def build_core_dataset(
  df_ratings_clean,
  processed_dir,
  min_reviews=2,
  save_name="ratings_core_for_graph.csv",
):
  # Total counts before filtering
  total_ratings = len(df_ratings_clean)
  total_users = df_ratings_clean["user_id"].nunique()
  total_books = df_ratings_clean["book_id"].nunique()
  # Compute the number of reviews per user and per book
  user_counts = df_ratings_clean["user_id"].value_counts()
  book_counts = df_ratings_clean["book_id"].value_counts()
  # Select users and books with at least min_reviews reviews
  active_users = user_counts[user_counts >= min_reviews].index
  active_books = book_counts[book_counts >= min_reviews].index

  # Filter the original dataframe to keep only active users and books
  df_core = df_ratings_clean[
      df_ratings_clean["user_id"].isin(active_users)
      & df_ratings_clean["book_id"].isin(active_books)
  ].copy()

  # Counts after filtering
  core_ratings = len(df_core)
  core_users = df_core["user_id"].nunique()
  core_books = df_core["book_id"].nunique()
  # Print some information about the core dataset
  print("Shape df_core:", df_core.shape)
  print("Distinct users in core:", core_users)
  print("Distinct books in core:", core_books)
  print(df_core.head())

  # Build the full path for the output file and save the core dataset
  core_path = os.path.join(processed_dir, save_name)
  df_core.to_csv(core_path, index=False)
  print(f"Core dataset saved in: {core_path}")

  return df_core