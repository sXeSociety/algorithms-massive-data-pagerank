import pandas as pd

# Compute and print basic statistics about users, books and reviews.
def describe_reviews(df_ratings_clean):

  # Compute the number of distinct users and books
  n_users = df_ratings_clean["user_id"].nunique()
  n_books = df_ratings_clean["book_id"].nunique()
  print(f"Distinct users: {n_users}")
  print(f"Distinct books: {n_books}")

  # Compute the distribution of reviews per user
  user_reviews = df_ratings_clean["user_id"].value_counts()
  print("\nReviews per user:")
  print(f"min: {user_reviews.min()}")
  print(f"median: {user_reviews.median()}")
  print(f"mean: {user_reviews.mean():.2f}")
  print(f"max: {user_reviews.max()}")

  # Compute the distribution of reviews per book
  book_reviews = df_ratings_clean["book_id"].value_counts()
  print("\nReviews per book:")
  print(f"min: {book_reviews.min()}")
  print(f"median: {book_reviews.median()}")
  print(f"mean: {book_reviews.mean():.2f}")
  print(f"max: {book_reviews.max()}")

  # Count how many users and books have at least two reviews
  active_users_count = (user_reviews >= 2).sum()
  active_books_count = (book_reviews >= 2).sum()
  print(f"\nUsers with >= 2 reviews: {active_users_count}")
  print(f"Books with >= 2 reviews: {active_books_count}")

  # Return all the useful objects in case we want to reuse them
  stats = {
      "n_users": n_users,
      "n_books": n_books,
      "user_reviews": user_reviews,
      "book_reviews": book_reviews,
      "active_users_ge2": active_users_count,
      "active_books_ge2": active_books_count,
  }

  return stats