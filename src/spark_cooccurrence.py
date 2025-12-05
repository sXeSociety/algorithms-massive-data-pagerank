import os
from itertools import combinations
import pandas as pd
from pyspark.sql import SparkSession

# Create and return a Spark session
def create_spark_session(app_name="BookCooccurrenceSparkBig", master="local[*]"):
    spark = (
      SparkSession.builder
      .appName(app_name)
      .master(master)
      .getOrCreate()
    )
    return spark

# Build book cooccurrence edges using Spark
def build_book_cooccurrence_edges_spark(
  spark,
  processed_dir,
  ratings_indexed_filename="ratings_core_big_indexed.csv",
  max_books_per_user=50,
  min_weight=1,
): 
  path = os.path.join(processed_dir, ratings_indexed_filename)
  print("\nSpark loading indexed ratings from:", path)

  df_indexed_big_spark = (
      spark.read
      .option("header", True)
      .option("inferSchema", True)
      .csv(path)
  )

  print("\nSpark indexed ratings schema:")
  df_indexed_big_spark.printSchema()
  # Keep only user and book indices
  df_pairs = df_indexed_big_spark.select("user_idx", "book_idx")
  # Map to pairs of integers
  ratings_pairs_rdd = df_pairs.rdd.map(
    lambda row: (int(row["user_idx"]), int(row["book_idx"])))
  print("\nSpark example user book pairs:")
  print(ratings_pairs_rdd.take(5))

  # Group books by user
  user_books_rdd = ratings_pairs_rdd.groupByKey()
  print("\nSpark example user with books:")
  for ub in user_books_rdd.take(3):
      print(ub)
  
  # For each user emit all book pairs with count one
  def user_to_book_pairs(user_books):
      user_idx, books_iter = user_books
      books = sorted(set(books_iter))
      if len(books) < 2:
        return []
      if max_books_per_user is not None and len(books) > max_books_per_user:
        return []
      pairs = [((b1, b2), 1) for b1, b2 in combinations(books, 2)]
      return pairs
  
  # create pairs and count cooccurrences
  book_pair_count_rdd = user_books_rdd.flatMap(user_to_book_pairs)
  print("\nSpark example book pair with count one:")
  print(book_pair_count_rdd.take(5))

  cooccurrence_rdd = book_pair_count_rdd.reduceByKey(lambda x, y: x + y)
  print("\nSpark example book pair with weight:")
  print(cooccurrence_rdd.take(5))

  # Map to triplets for DataFrame
  edges_triplets_rdd = cooccurrence_rdd.map(
      lambda pair_weight: (
          int(pair_weight[0][0]),
          int(pair_weight[0][1]),
          int(pair_weight[1]),
      )
  )

  edges_big_spark = edges_triplets_rdd.toDF(
      ["src_book_idx", "dst_book_idx", "weight"])
  
  print("\nSpark edges schema:")
  edges_big_spark.printSchema()
  print("\nSpark first edges:")
  edges_big_spark.show(10)
  print("\nSpark number of edges before filter:", edges_big_spark.count())

  # Convert to pandas for comparison
  edges_df_big_spark = edges_big_spark.toPandas()
  print("\nSpark edges DataFrame shape:", edges_df_big_spark.shape)
  return edges_df_big_spark

# Compare Python and Spark edge lists
def compare_edges_python_spark(edges_df_python, edges_df_spark):
  common_cols = ["src_book_idx", "dst_book_idx", "weight"]
  edges_py_sorted = (
      edges_df_python[common_cols]
      .sort_values(common_cols)
      .reset_index(drop=True)
  )
  edges_sp_sorted = (
      edges_df_spark[common_cols]
      .sort_values(common_cols)
      .reset_index(drop=True)
  )

  same_shape = edges_py_sorted.shape == edges_sp_sorted.shape
  same_content = edges_py_sorted.equals(edges_sp_sorted)
  print("\nCompare python and spark edges")
  print("same shape:", same_shape)
  print("same content:", same_content)
  return same_shape and same_content