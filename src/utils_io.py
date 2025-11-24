import os
<<<<<<< HEAD
def ensure_dirs(dirs):
    """
    Create all directories in the list if they do not exist.

    Args:
        dirs (list of str): List of directory paths.
    """
    for d in dirs:
        # Check if the directory does not exist
        if not os.path.exists(d):
            # Create the directory and all missing parent directories
            os.makedirs(d, exist_ok=True)
            print(f"Created directory: {d}")
        else:
            # Directory already exists, nothing to do
            print(f"Directory already exists: {d}")
=======

# Create all the directories if they do not exist.
def create_dirs(dirs):
  for d in dirs:
    if not os.path.exists(d):
      os.makedirs(d, exist_ok=True)
      print(f"Created directory: {d}")
    else:
      print(f"Directory already exists: {d}")
>>>>>>> 40f72b54e3d9773c9bc3d0b9047b0dcd1fd4762f
