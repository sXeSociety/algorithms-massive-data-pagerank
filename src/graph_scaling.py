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

  results = []

  for cfg in configs:
    name = cfg["name"]
    max_users = cfg["max_users"]
    print(f"\nScaling config: {name} with max_users={max_users}")
    # Build a subset of the core dataset
    subset_filename = f"ratings_core_{name}_for_graph.csv"
    df_subset = build_core_subset(
      df_core=df_core,
      processed_dir=processed_dir,
      max_users=max_users,
      save_name=subset_filename,
    )

    # Build integer mappings for users and books
    user_mapping_name = f"user_id_mapping_{name}.csv"
    book_mapping_name = f"book_id_mapping_{name}.csv"
    indexed_ratings_name = f"ratings_core_{name}_indexed.csv"

    user_mapping, book_mapping, df_indexed = build_id_mappings(
      df_core_small=df_subset,
      processed_dir=processed_dir,
      user_mapping_name=user_mapping_name,
      book_mapping_name=book_mapping_name,
      ratings_indexed_name=indexed_ratings_name,
    )

    # Build the book cooccurrence edge list and measure time
    edges_filename = f"edges_books_core_{name}.csv"
    t_graph_start = time.perf_counter()
    edges_df = build_book_cooccurrence_edges(
      df_indexed=df_indexed,
      processed_dir=processed_dir,
      save_name=edges_filename,
      max_books_per_user=max_books_per_user,
      min_weight=min_weight,
    )
    t_graph_end = time.perf_counter()
    graph_time = t_graph_end - t_graph_start

    if run_sanity_checks_flag:
      run_all_sanity_checks(
        df_core_small=df_subset,
        df_indexed=df_indexed,
        user_mapping=user_mapping,
        book_mapping=book_mapping,
        edges_df=edges_df,
      )
    
    num_nodes = len(book_mapping)
    num_edges = len(edges_df)
    print(f"Config {name} nodes: {num_nodes}")
    print(f"Config {name} edges: {num_edges}")
    print(f"Config {name} graph build time seconds: {graph_time:.4f}")

    # Build directed edges from the undirected cooccurrence graph
    src_nodes = np.concatenate(
      [edges_df["src_book_idx"].values,
      edges_df["dst_book_idx"].values,])
    dst_nodes = np.concatenate(
      [edges_df["dst_book_idx"].values,
      edges_df["src_book_idx"].values,])

    # Run pagerank and measure time
    t_pr_start = time.perf_counter()
    ranks = pagerank_power_iteration(
      num_nodes=num_nodes,
      src_nodes=src_nodes,
      dst_nodes=dst_nodes,
      damping=damping,
      tol=tol,
      max_iter=max_iter,
      verbose=verbose_pagerank,)
    t_pr_end = time.perf_counter()
    pagerank_time = t_pr_end - t_pr_start
    print(f"Config {name} pagerank time seconds: {pagerank_time:.4f}")
    print(f"Config {name} pagerank sum: {ranks.sum():.6f}")

    # Collect summary stats for this config
    summary = {"config_name": name,
            "max_users": max_users,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "graph_build_time_sec": graph_time,
            "pagerank_time_sec": pagerank_time,
        }
    results.append(summary)

    # Build summary DataFrame
    df_results = (
        pd.DataFrame(results)
        .sort_values("max_users")
        .reset_index(drop=True)
    )

    print("\nScaling experiments summary:")
    print(df_results)

    if save_results:
        path = os.path.join(processed_dir, results_filename)
        df_results.to_csv(path, index=False)
        print("Saved scaling summary to:", path)

    return df_results