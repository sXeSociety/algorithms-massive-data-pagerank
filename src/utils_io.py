import os

# Create all the directories if they do not exist.
def create_dirs(dirs):
  for d in dirs:
    if not os.path.exists(d):
      os.makedirs(d, exist_ok=True)
      print(f"Created directory: {d}")
    else:
      print(f"Directory already exists: {d}")