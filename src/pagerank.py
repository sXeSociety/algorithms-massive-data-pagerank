import numpy as np

# Compute PageRank scores using the power iteration method
def pagerank_power_iteration(
    num_nodes,
    src_nodes,
    dst_nodes,
    damping=0.85,
    tol=1e-6,
    max_iter=100,
    verbose=False,
):
  # Convert src_nodes and dst_nodes to numpy arrays of type int
  src_nodes = np.asarray(src_nodes, dtype=int)
  dst_nodes = np.asarray(dst_nodes, dtype=int)

  # Basic checks on the inputs
  if src_nodes.shape[0] != dst_nodes.shape[0]:
    raise ValueError("src_nodes and dst_nodes must have the same length.")
  
  if src_nodes.size == 0:
      # Edge case: no edges at all, return uniform distribution
      if verbose:
        print("[pagerank] No edges found. Returning uniform ranks.")
      return np.ones(num_nodes, dtype=float) / num_nodes

  if src_nodes.max() >= num_nodes or dst_nodes.max() >= num_nodes:
      raise ValueError("Node indices in src_nodes/dst_nodes must be < num_nodes.")

  # Compute out-degree for each node (number of outgoing edges)
  out_degree = np.bincount(src_nodes, minlength=num_nodes).astype(float)
  # Identify dangling nodes (nodes with no outgoing edges)
  dangling_mask = (out_degree == 0)
  # Initialize PageRank vector with uniform distribution
  ranks = np.ones(num_nodes, dtype=float) / num_nodes

  # Precompute teleportation term (uniform teleport)
  teleport = (1.0 - damping) / num_nodes

  if verbose:
      print("[pagerank] Starting power iteration...")
      print(f"[pagerank] num_nodes = {num_nodes}")
      print(f"[pagerank] damping   = {damping}")
      print(f"[pagerank] tol       = {tol}")
      print(f"[pagerank] max_iter  = {max_iter}")
      print(f"[pagerank] teleport term = {teleport}")

  # Power iteration loop
  for it in range(1, max_iter + 1):
      # Keep a copy of the current ranks
      ranks_old = ranks

      # Contribution from dangling nodes: their rank is redistributed uniformly
      dangling_rank = ranks_old[dangling_mask].sum()
      dangling_contrib = damping * dangling_rank / num_nodes
      # Contribution passed along the edges
      # Each outgoing edge from node u carries ranks_old[u] / out_degree[u]
      valid_src_mask = (out_degree[src_nodes] > 0)
      contrib_weights = np.zeros_like(src_nodes, dtype=float)
      contrib_weights[valid_src_mask] = (
          ranks_old[src_nodes[valid_src_mask]] /
          out_degree[src_nodes[valid_src_mask]]
      )

      # Sum contributions for each destination node
      link_contrib = np.bincount(
          dst_nodes,
          weights=contrib_weights,
          minlength=num_nodes,
      )

      # Apply damping factor to the contribution coming from links
      link_contrib *= damping
      # Combine teleportation, dangling contribution and link contribution
      ranks = teleport + dangling_contrib + link_contrib
      # Normalize to ensure the ranks sum to 1 (numerical stability)
      ranks_sum = ranks.sum()
      if ranks_sum > 0:
          ranks /= ranks_sum

      # Compute L1 difference between consecutive iterations
      diff = np.abs(ranks - ranks_old).sum()

      if verbose:
          print(f"[pagerank] Iteration {it:3d} – diff = {diff:.6e}")

      # Check convergence
      if diff < tol:
        if verbose:
              print(f"[pagerank] Converged in {it} iterations.")
        break
  else:
      # If we exit the loop without break (no convergence before max_iter)
      if verbose:
          print(
              f"[pagerank] Reached max_iter = {max_iter} "
              f"with diff = {diff:.6e}"
          )

  return ranks