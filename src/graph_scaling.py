import os
import time

import numpy as np
import pandas as pd

from src.preprocessing import build_core_subset
from src.mapping_ids import build_id_mappings
from src.graph_construction import build_book_cooccurrence_edges
from src.pagerank import pagerank_power_iteration
from src.debug_utils import run_all_sanity_checks

# Run graph and pagerank scaling experiments for a list of configs.
def run_scaling_experiments(
    df_core,
    processed_dir,
    configs,
    max_books_per_user=50,
    min_weight=1,
    damping=0.85,
    tol=1e-6,
    max_iter=100,
    verbose_pagerank=False,
    run_sanity_checks_flag=True,
    save_results=True,
    results_filename="graph_scaling_summary.csv",
):

  records = []

  for cfg in configs:
    config_name = cfg["name"]
    max_users = cfg["max_users"]
    print(f"\n[scaling] running config {config_name} with max_users={max_users}")

    # Build core subset
    subset_name = f"ratings_core_{config_name}_for_graph.csv"
    df_core_sub = build_core_subset(
      df_core=df_core,
      processed_dir=processed_dir,
      max_users=max_users,
      save_name=subset_name,)
    
    # Build index mappings
    user_mapping_name = f"user_id_mapping_{config_name}.csv"
    book_mapping_name = f"book_id_mapping_{config_name}.csv"
    ratings_indexed_name = f"ratings_core_{config_name}_indexed.csv"

    user_mapping, book_mapping, df_indexed = build_id_mappings(
      df_core_small=df_core_sub,
      processed_dir=processed_dir,
      user_mapping_name=user_mapping_name,
      book_mapping_name=book_mapping_name,
      ratings_indexed_name=ratings_indexed_name,)
    
    # Build cooccurrence graph and measure time
    t_graph_start = time.perf_counter()
    edges_df = build_book_cooccurrence_edges(
      df_indexed=df_indexed,
      processed_dir=processed_dir,
      save_name=f"edges_books_core_{config_name}.csv",
      max_books_per_user=max_books_per_user,
      min_weight=min_weight,)
    t_graph_end = time.perf_counter()
    graph_build_time = t_graph_end - t_graph_start

    num_nodes = len(book_mapping)
    num_edges = len(edges_df)

    print(
      f"[scaling] config {config_name} "
      f"nodes={num_nodes} edges={num_edges} "
      f"graph_build_time={graph_build_time:.4f} seconds"
    )
    pagerank_time = float("nan")

    if run_sanity_checks_flag:
      run_all_sanity_checks(
        df_core_small=df_core_sub,
        df_indexed=df_indexed,
        user_mapping=user_mapping,
        book_mapping=book_mapping,
        edges_df=edges_df,)
    
    # If no edges, skip PageRank but still record information
    if num_edges == 0 and num_nodes >= 2:
      print(f"[scaling] config {config_name} has zero edges, skipping pagerank")
      pagerank_time=np.nan
    else:
      # Build directed edges
      src_nodes = np.concatenate(
        [edges_df["src_book_idx"].values, edges_df["dst_book_idx"].values])
      dst_nodes = np.concatenate(
        [edges_df["dst_book_idx"].values, edges_df["src_book_idx"].values])
      
      # Run PageRank and measure time
      t_pr_start = time.perf_counter()
      _ = pagerank_power_iteration(
        num_nodes=num_nodes,
        src_nodes=src_nodes,
        dst_nodes=dst_nodes,
        damping=damping,
        tol=tol,
        max_iter=max_iter,
        verbose=verbose_pagerank,)
      t_pr_end = time.perf_counter()
      pagerank_time = t_pr_end - t_pr_start
      
      print(
        f"[scaling] config {config_name} "
        f"pagerank_time={pagerank_time:.4f} seconds"
      )
    
    record = {
      "config_name": config_name,
      "max_users": max_users,
      "num_nodes": num_nodes,
      "num_edges": num_edges,
      "graph_build_time_sec": graph_build_time,
      "pagerank_time_sec": pagerank_time,}
    records.append(record)

  df_scaling = pd.DataFrame.from_records(records)
  if save_results:
    os.makedirs(processed_dir, exist_ok=True)
    results_path = os.path.join(processed_dir, results_filename)
    df_scaling.to_csv(results_path, index=False)
    print(f"[scaling] saved results to {results_path}")
  
  return df_scaling
