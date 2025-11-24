import os
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
