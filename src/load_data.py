import os
import numpy as np
import pandas as pd
from src.utils_io import ensure_dirs

# Return the expected path for the ratingS CSV file.
def ratings_file_path(raw_dir):
  return os.path.join(raw_dir, "Books_rating.csv")

# Download the dataset into raw_dir only if the ratings file does not already exist.
def download_dataset(raw_dir, kaggle_dataset):
  ensure_dirs([raw_dir])
  # Check if the file is present
  ratings_path = ratings_file_path(raw_dir)
  if os.path.exists(ratings_path):
    print("Dataset already present in raw_dir, skipping download.")
    return
  print("Dataset not found in raw_dir, downloading from Kaggle.")

  # Build the Kaggle CLI command and then execute it
  cmd = f'kaggle datasets download -d {kaggle_dataset} -p {raw_dir} --unzip'
  exit_code = os.system(cmd)

  # Check if the command completed successfully
  if exit_code != 0:
      raise RuntimeError(
          "Error while downloading dataset from Kaggle. "
          "Check Kaggle CLI installation and credentials."
      )

  print("Download extraction completed.")
  print("Files now in raw_dir:")
  for f in os.listdir(raw_dir):
      print(" -", f)

  # Final check that the ratings file is present
  if not os.path.exists(ratings_path):
      raise FileNotFoundError(f"Expected ratings file not found at {ratings_path}") 

# Load the ratings CSV file, optionally create a subsample, 
# rename the columns and save the result into the processed directory
def load_ratings (
  raw_dir, 
  processed_dir,
  use_subsample = True,
  subsample_fraction=0.05,
  seed = 42,
  save_subsample_name = "ratings_subsample.csv",
  save_clean_name = "ratings_subsample_clean.csv",
):

  # Make sure the processed_dir exists
  ensure_dirs([processed_dir])
  # Build the path to the ratings file
  ratings_path = ratings_file_path(raw_dir)
  # Build the path to the cleaned file in processed_dir
  clean_path = os.path.join(processed_dir, save_clean_name)
  # If we already have the cleaned file, just load it and return it
  if os.path.exists(clean_path):
      print(f"Found existing cleaned ratings at: {clean_path}")
      df_ratings_clean = pd.read_csv(clean_path)
      print("Shape df_ratings_clean (loaded from disk):", df_ratings_clean.shape)
      print(df_ratings_clean.head())
      return df_ratings_clean
  # Check that the ratings file exists
  if not os.path.exists(ratings_path):
      raise FileNotFoundError(f"ERROR: ratings file not found at {ratings_path}")
  # Set the random seed for reproducibility
  np.random.seed(seed)

  # Load the full dataset
  if use_subsample:
    print("Using subsample mode.")
    df_ratings = pd.read_csv(ratings_path)
    print(f"Full dataset shape before subsample: {df_ratings.shape}")
  
    df_ratings = df_ratings.sample(
          frac=subsample_fraction,
          random_state=seed,
    )
    print(f"Subsampled dataset shape: {df_ratings.shape}")

    # Save the sampled dataset into the processed directory
    subsample_path = os.path.join(processed_dir, save_subsample_name)
    df_ratings.to_csv(subsample_path, index = False)
    print(f"Subsample saved to: {subsample_path}")
  
  else:
    print("Loading full dataset (no subsample)...")
    df_ratings = pd.read_csv(ratings_path)
    print(f"Full dataset shape: {df_ratings.shape}")
  
  # Rename the columns to more understandable names and keep only what is useful
  df_ratings_clean = df_ratings.rename(
      columns={
          "User_id": "user_id",
          "Id": "book_id",
          "review/score": "rating",
      }
  )[["user_id", "book_id", "rating"]]

  print("Shape df_ratings_clean:", df_ratings_clean.shape)
  print(df_ratings_clean.head())

  # Save the cleaned dataset into the processed directory
  clean_path = os.path.join(processed_dir, save_clean_name)
  df_ratings_clean.to_csv(clean_path, index=False)
  print(f"Clean subsample saved in: {clean_path}")

  return df_ratings_clean