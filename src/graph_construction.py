import os
import itertools
from collections import Counter
import pandas as pd
from src.utils_io import ensure_dirs

# Build an undirected co-occurrence graph of books
def build_book_cooccurrence_edges(
  df_indexed,
  processed_dir,
  save_name="edges_books_core_small.csv",
  max_books_per_user=None,
  min_weight=1,
):
  ensure_dirs([processed_dir])
  print("\n[build_book_cooccurrence_edges] Building book co-occurrence graph...")

  # Counter to store edge weights (i,j)
  # We will always store edges with (i < j) to represent an undirected edge.
  edge_counter = Counter()

  # Group by user_idx to get, for each user, the list of books they reviewed
  for user_idx, group in df_indexed.groupby("user_idx"):
    # Get the list of unique book indices for this user
    books = group["book_idx"].unique()

    # If the user has reviewed fewer than 2 books, they do not create any edge
    if len (books) < 2:
      continue

    # Skip users with too many books
    if max_books_per_user is not None and len(books) > max_books_per_user:
      continue
    
    # Sort the book indices to have a deterministic order
    books = sorted(books)

    # Generate all combinations of 2 different books
    for i,j in itertools.combinations(books,2):
      # (i,j) is already in increasing order because we sorted books
      edge_counter[(i,j)] += 1
    
  print(f"[build_book_cooccurrence_edges] Number of distinct edges: {len(edge_counter)}")

  # Convert the counter into a DataFrame edge list
  if edge_counter:
    src_nodes = []
    dst_nodes = []
    weights = []

    for (i,j), w in edge_counter.items():
      src_nodes.append(i)
      dst_nodes.append(j)
      weights.append(w)
    
    edges_df = pd.DataFrame ({
      "src_book_idx": src_nodes,
      "dst_book_idx": dst_nodes,
      "weight": weights,
    })
  
  else:
    # Edge case: no edges created
    edges_df = pd.DataFrame(
      columns = ["src_book_idx", "dst_book_idx", "weight"]
    )
  
  # Filter edges by minimum weight
  if min_weight is not None and min_weight > 1 and not edges_df.empty:
    before = len(edges_df)
    edges_df = edges_df[edges_df["weight"] >= min_weight].copy()
    after = len(edges_df)
    print(
      f"[build_book_cooccurrence_edges] Filtered edges with weight < {min_weight}: "
      f"{before} -> {after}"
    )
  
  print("\n[build_book_cooccurrence_edges] Edge list (first rows):")
  print(edges_df.head())

  # Save the edge list to CSV
  edges_path = os.path.join(processed_dir, save_name)
  edges_df.to_csv(edges_path, index=False)
  print(f"[build_book_cooccurrence_edges] Edge list saved in: {edges_path}")

  return edges_df