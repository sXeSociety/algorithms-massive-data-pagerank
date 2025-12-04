import numpy as np

# Compute PageRank scores using the power iteration method
def pagerank_power_iteration(
  num_nodes,
  src_nodes,
  dst_nodes,
  edge_weights=None,
  damping=0.85,
  tol=1e-6,
  max_iter=100,
  verbose=False,
  teleport_vector=None
):
  # Convert src_nodes and dst_nodes to numpy arrays of type int
  src_nodes = np.asarray(src_nodes, dtype=int)
  dst_nodes = np.asarray(dst_nodes, dtype=int)

  # Basic checks on the inputs
  if src_nodes.shape[0] != dst_nodes.shape[0]:
    raise ValueError("src_nodes and dst_nodes must have the same length.")
  
  num_edges = src_nodes.size
  if src_nodes.size == 0:
    # Edge case: no edges at all, return uniform distribution
    if verbose:
      print("[pagerank] No edges found. Returning uniform ranks.")
    return np.ones(num_nodes, dtype=float)/num_nodes

  if src_nodes.max() >= num_nodes or dst_nodes.max() >= num_nodes:
    raise ValueError ("Node indices in src_nodes/dst_nodes must be < num_nodes.")

  # If provided, we enable the weighted mode
  if edge_weights is not None:
    edge_weights = np.asarray(edge_weights, dtype=float)
    if edge_weights.shape[0] != num_edges:
      raise ValueError (
        "edge_weights must have the same length as src_nodes/dst_nodes."
      )
    use_weights = True
  else:
    use_weights = False
  
  # Precompute the "out measure" for each node:
  if use_weights:
    out_measure = np.bincount(
      src_nodes,
      weights = edge_weights,
      minlength = num_nodes,
    ).astype(float)
    if verbose: 
      print("[pagerank] Weighted mode enabled (using edge_weights).")
  
  else:
    out_measure = np.bincount(src_nodes, minlength=num_nodes).astype(float)
    if verbose:
      print("[pagerank] Unweighted mode (using out-degrees).")

  # Dangling nodes are those with no outgoing edges (or zero out-strength)
  dangling_mask = (out_measure == 0)
  # Initialize PageRank vector with uniform distribution
  ranks = np.ones(num_nodes, dtype=float) / num_nodes

  # Prepare teleportation base term
  if teleport_vector is None:
      # Uniform teleportation over all nodes
      base_teleport = np.full(
          shape=num_nodes,
          fill_value=(1.0 - damping) / num_nodes,
          dtype=float,
      )
  else:
      tv = np.asarray(teleport_vector, dtype=float)
      if tv.shape[0] != num_nodes:
        raise ValueError("teleport_vector must have length num_nodes.")
      total_tv = tv.sum()
      if total_tv <= 0:
        raise ValueError("teleport_vector must have a positive sum.")
      # Normalize teleport_vector to sum to 1, then scale by (1 - damping)
      tv = tv / total_tv
      base_teleport = (1.0 - damping) * tv
  
  if verbose:
        print("[pagerank] Starting power iteration...")
        print(f"[pagerank] num_nodes = {num_nodes}")
        print(f"[pagerank] damping = {damping}")
        print(f"[pagerank] tol = {tol}")
        print(f"[pagerank] max_iter = {max_iter}")

  # Power iteration loop
  for it in range(1, max_iter + 1):
      # Keep a copy of the current ranks
      ranks_old = ranks
      # Contribution from dangling nodes: their rank is redistributed uniformly
      dangling_rank = ranks_old[dangling_mask].sum()
      dangling_contrib = damping * dangling_rank / num_nodes
      # Contribution passed along the edges
      contrib_weights = np.zeros(num_edges, dtype=float)
      # Only edges whose source has positive out_measure contribute
      valid_src_mask = (out_measure[src_nodes] > 0)

      if use_weights:
          # Weighted mode: each edge (u -> v) gets
          # rank[u] * w_uv / out_strength[u]
          contrib_weights[valid_src_mask] = (
              ranks_old[src_nodes[valid_src_mask]]
              * edge_weights[valid_src_mask]
              / out_measure[src_nodes[valid_src_mask]])
      else:
          # Unweighted mode: each edge (u -> v) gets rank[u] / out_degree[u]
          contrib_weights[valid_src_mask] = (
              ranks_old[src_nodes[valid_src_mask]]
              / out_measure[src_nodes[valid_src_mask]])

      link_contrib = np.bincount(
          dst_nodes,
          weights=contrib_weights,
          minlength=num_nodes,
      )

      # Apply damping factor to the contribution coming from links
      link_contrib *= damping
      # Combine teleportation, dangling contribution and link contribution
      ranks = base_teleport + dangling_contrib + link_contrib
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
