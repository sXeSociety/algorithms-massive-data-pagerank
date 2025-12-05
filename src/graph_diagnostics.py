import numpy as np
import pandas as pd

# Print summary statistics for edge weights
def summarize_edge_weights(weights):
  weights = pd.Series(weights)
  total = len(weights)
  print("\nEdge weight summary")
  print(f"Number of edges: {total}")

  # Counts of exact weights
  for thr in [1, 2, 3, 5, 10]:
    count_eq = (weights == thr).sum()
    share_eq = count_eq / total if total > 0 else np.nan
    print(f"  weight == {thr:2d}: {count_eq:7d} ({share_eq:6.2%})")
  
  # Counts of weights above thresholds
  for thr in [2, 3, 5, 10]:
    count_ge = (weights >= thr).sum()
    share_ge = count_ge / total if total > 0 else np.nan
    print(f"  weight >= {thr:2d}: {count_ge:7d} ({share_ge:6.2%})")
  
  # Selected quantities
  quantiles = weights.quantile([0.50, 0.75, 0.90, 0.95, 0.99])
  print("\n  Selected quantiles:")
  for q, val in quantiles.items():
    print(f"    q={q:4.2f}: {val:.3f}")
  
# Compute degree, strength, average weight per node.
def compute_node_statistics(edges_df):
  # Degree from src and dst occurrences
  degree = pd.concat([
    edges_df["src_book_idx"],
    edges_df["dst_book_idx"],
  ]).value_counts().rename("degree")

  degree = degree.reset_index().rename(columns={"index": "book_idx"})

  # strength = sum of weights on incident edges
  strength_src = edges_df.groupby("src_book_idx")["weight"].sum()
  strength_dst = edges_df.groupby("dst_book_idx")["weight"].sum()
  strength = strength_src.add(strength_dst, fill_value=0)
  strength_df = strength.rename("strength").reset_index().rename(
    columns={"index": "book_idx"}
  )

  # Merge into one df
  df_stats = degree.merge(strength_df, on="book_idx", how="left")
  df_stats["strength"] = df_stats["strength"].fillna(0)
  # Average weight = strength / degree
  df_stats["avg_weight"] = np.where(
    df_stats["degree"] > 0,
    df_stats["strength"] / df_stats["degree"],
    np.nan,
  )
  return df_stats

# Select nodes with degree >= min_degree and highest avg_weight
def top_nodes_by_avg_weight(df_stats, min_degree=3, top_k=20):
  df = df_stats[df_stats["degree"] >= min_degree]
  return df.sort_values("avg_weight", ascending=False).head(top_k)