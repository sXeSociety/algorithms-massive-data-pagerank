import os
import numpy as np
import pandas as pd

from src.utils_io import ensure_dirs

# Build integer index mappings for users and books starting from a small core rating database
def build_id_mappings(
  df_core_small,
  processed_dir,
  user_mapping_name="user_id_mapping_small.csv",
  book_mapping_name="book_id_mapping_small.csv",
  ratings_indexed_name="ratings_core_mapping_small.csv",
):
  # Make sure the processed directory exists
  ensure_dirs([processed_dir])
  print("\n[build_id_mappings] Creating user/book integer index mappings...")

  # Build the user_id - user_idx mapping (sorted for reproducibility)
  unique_users = np.sort(df_core_small["user_id"].unique())
  user_mapping = pd.DataFrame({
    "user_id": unique_users,
    "user_idx": np.arange(len(unique_users), dtype=int),
  })

  # Build the book_id - book_idx mapping
  unique_books = np.sort(df_core_small["book_id"].unique())
  book_mapping = pd.DataFrame({
    "book_id": unique_books,
    "book_idx": np.arange(len(unique_books), dtype=int),
  })

  # Merge these mappings into the dataset
  df_indexed = df_core_small.merge(user_mapping, on="user_id", how="left")
  df_indexed = df_indexed.merge(book_mapping, on="book_id", how="left")
  # There must be no missing indices
  if df_indexed["user_idx"].isna().any():
      raise ValueError("[build_id_mappings] Missing user_idx after merge.")
  if df_indexed["book_idx"].isna().any():
      raise ValueError("[build_id_mappings] Missing book_idx after merge.")
  
  # After the merge indices might be float
  df_indexed["user_idx"] = df_indexed["user_idx"].astype(int)
  df_indexed["book_idx"] = df_indexed["book_idx"].astype(int)

  # Save outputs
  user_mapping_path = os.path.join(processed_dir, user_mapping_name)
  book_mapping_path = os.path.join(processed_dir, book_mapping_name)
  indexed_ratings_path = os.path.join(processed_dir, ratings_indexed_name)

  user_mapping.to_csv(user_mapping_path, index=False)
  book_mapping.to_csv(book_mapping_path, index=False)
  df_indexed.to_csv(indexed_ratings_path, index=False)

  print(f"[build_id_mappings] Saved user mapping:   {user_mapping_path}")
  print(f"[build_id_mappings] Saved book mapping:   {book_mapping_path}")
  print(f"[build_id_mappings] Saved indexed ratings: {indexed_ratings_path}")

  return user_mapping, book_mapping, df_indexed