import itertools
import numpy as np
import pandas as pd

# Check that mapping cardinalities match the small core dataset
def _sanity_check_mappings(df_core_small, user_mapping, book_mapping):
  # Number of unique users and books in the small core dataset
  n_users_core_small = df_core_small["user_id"].nunique()
  n_books_core_small = df_core_small["book_id"].nunique()
  print("\n[Sanity 1] Unique counts in df_core_small:")
  print(f"Users in df_core_small: {n_users_core_small}")
  print(f"Books in df_core_small: {n_books_core_small}")

  # Number of rows in the mapping tables
  n_users_mapping = len(user_mapping)
  n_books_mapping = len(book_mapping)
  print("\n[Sanity 1] Mapping sizes:")
  print(f"Rows in user_mapping: {n_users_mapping}")
  print(f"Rows in book_mapping: {n_books_mapping}")

  # Each user/book must appear in the mapping
  assert n_users_core_small == n_users_mapping, ("[Sanity 1] Mismatch between unique users and user_mapping rows.")
  assert n_books_core_small == n_books_mapping, ("[Sanity 1] Mismatch between unique books and book_mapping rows.")
  print("[Sanity 1] OK: mapping cardinalities match the small core dataset.")

# Check basic graph size: number of nodes and edges
def _sanity_check_graph_size(book_mapping, edges_df):
  n_nodes = len(book_mapping)
  n_edges = len(edges_df)
  print("\n[Sanity 2] Graph size:")
  print(f"Number of nodes (books): {n_nodes}")
  print(f"Number of edges:         {n_edges}")

# Inspect edge list structure and weight range
def _sanity_check_edges(edges_df):
  print("\n[Sanity 3] Edge list head:")
  print(edges_df.head())
  print("\n[Sanity 3] Edge list dtypes:")
  print(edges_df.dtypes)

  # Check that weights are >= 1, as they represent co-occurrence counts
  min_weight_observed = edges_df["weight"].min()
  max_weight_observed = edges_df["weight"].max()
  print("\n[Sanity 3] Edge weight range:")
  print(f"min weight: {min_weight_observed}")
  print(f"max weight: {max_weight_observed}")

  assert min_weight_observed >= 1, ("[Sanity 3] Found an edge with weight < 1.")
  print("[Sanity 3] OK: weights look consistent (>= 1).")

# Manually verify that all pairs of books for an example user appear in the edge list
def _sanity_check_user_pairs(df_indexed, edges_df, max_books_for_example=6):
  # Count how many rows each user contributes
  user_book_counts = df_indexed["user_idx"].value_counts()

  # Select one user with at least 2 and at most max_books_for_example books
  candidate_users = user_book_counts[
    user_book_counts.between(2, max_books_for_example)
  ]
  if candidate_users.empty:
    print("\n[Sanity 4] No user found with a small number of books "
      f"(<= {max_books_for_example}). Skipping manual pair check.")
    return
  
  example_user_idx = candidate_users.index[0]
  print(f"\n[Sanity 4] Example user_idx selected: {example_user_idx}")

  # Get the list of unique books reviewed by this user
  user_books = df_indexed.loc[
    df_indexed["user_idx"] == example_user_idx, "book_idx"
  ].unique()
  user_books = sorted(user_books)
  print(f"[Sanity 4] Books reviewed by this user: {user_books}")

  # Generate all pairs of books for this user
  expected_pairs = list(itertools.combinations(user_books, 2))
  print(f"[Sanity 4] Pairs that this user should contribute: {expected_pairs}")

  # For each pair, check that it appears in the edge list
  missing_pairs = []
  for i, j in expected_pairs:
    mask = (
      (edges_df["src_book_idx"] == i)
      & (edges_df["dst_book_idx"] == j)
    )
    if not mask.any():
      missing_pairs.append((i, j))

  if missing_pairs:
    print("\n[Sanity 4] WARNING: Some expected pairs are missing in the edge list:")
    print(missing_pairs)
  else:
    print("[Sanity 4] OK: all expected pairs for this user are present in the edge list.")

# Inspect the degree distribution (how many neighbors each book has)
def _sanity_check_degree_distribution(edges_df, top_k=10):
  # Build degree counts: each occurrence in src or dst contributes to the degree
  degree_series = pd.concat(
    [edges_df["src_book_idx"], edges_df["dst_book_idx"]]
  ).value_counts()

  print("\n[Sanity 5] Degree stats:")
  print(degree_series.describe())
  print(f"\n[Sanity 5] Top {top_k} nodes by degree:")
  print(degree_series.head(top_k))

# Run all these sanity checks in sequence
def run_all_sanity_checks(
  df_core_small,
  df_indexed,
  user_mapping,
  book_mapping,
  edges_df,
):
  _sanity_check_mappings(df_core_small, user_mapping, book_mapping)
  _sanity_check_graph_size(book_mapping, edges_df)
  _sanity_check_edges(edges_df)
  _sanity_check_user_pairs(df_indexed, edges_df)
  _sanity_check_degree_distribution(edges_df)