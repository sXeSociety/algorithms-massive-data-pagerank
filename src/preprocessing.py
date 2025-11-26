import os
import pandas as pd
import numpy as np

# df_core is the dataset I am going to use to create the graph and do PageRank
# The graph will have way less noise, it is going to be more connected and interesting
def build_core_dataset(
  df_ratings_clean,
  processed_dir,
  min_reviews=2,
  save_name="ratings_core_for_graph.csv",
):

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


# Build a smaller dataset for debugging and prototyping dataset construction.
def build_core_subset(
  df_core,
  processed_dir,
  max_users=2000,
  save_name="ratings_core_small_for_graph.csv",
):

  # Total counts before filtering
  total_ratings = len(df_core)
  total_users = df_core["user_id"].nunique()
  total_books = df_core["book_id"].nunique()
  
  df_subset = df_core.copy()

  # Select a subset of users
  if max_users is not None:
    # Get all distinct user_ids and sort them to have a deterministic selection
    unique_users = np.sort(df_core["user_id"].unique())
    # Take the first max_users
    selected_users = unique_users[:max_users]
    print(f"\n[build_core_subset] Limiting to first {len(selected_users)} users.")
    # Filter the dataframe
    df_subset = df_subset[df_subset["user_id"].isin(selected_users)]
  
  # Stats after filtering
  subset_ratings = len(df_subset)
  subset_users = df_subset["user_id"].nunique()
  print("\n[build_core_subset] Subset stats after filtering")
  print(f"Subset ratings: {subset_ratings}")
  print(f"Subset distinct users: {subset_users}")
  print(df_subset.head())

  # Build the full path for the output file and save the subset dataset
  subset_path = os.path.join(processed_dir, save_name)
  df_subset.to_csv(subset_path, index=False)
  print(f"Core subset dataset saved in: {subset_path}")

  return df_subset