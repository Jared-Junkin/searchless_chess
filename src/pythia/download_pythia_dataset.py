import logging
from huggingface_hub import HfApi, hf_hub_download
import os
# must set permissions for force the directory /searchless_chess to read and write to its immediate contents:,  sudo chmod u+w ~/searchless_chess/data
# must also configure timeout length like this:export HF_HUB_DOWNLOAD_TIMEOUT=100000
# otherwise you wil get timeout errors.
# Configure logging

logging.basicConfig(
    filename="download_pythia.log",  # Logfile name
    level=logging.INFO,  # Log everything at INFO level or higher
    format="%(asctime)s - %(levelname)s - %(message)s"  # Log format
)

# Repository details
repo_id = "EleutherAI/pile-standard-pythia-preshuffled"  # Repository name
repo_type = "dataset"  # Repository type
cache_dir = os.path.expanduser("~/searchless_chess/data/pythia/")

# Initialize API client
logging.info("Initializing API client")
api = HfApi()

# List all files in the repository
logging.info(f"Fetching file list from repository: {repo_id}")
try:
    files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    logging.info(f"Found {len(files)} files in the repository.")
except Exception as e:
    logging.error(f"Failed to fetch file list: {e}")
    raise

# Filter for `.bin` files
bin_files = [file for file in files if file.endswith(".bin")]
logging.info(f"Filtered {len(bin_files)} .bin files to download.")

# Download each `.bin` file
for bin_file in bin_files:
    try:
        logging.info(f"Starting download of {bin_file}")
        hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            cache_dir=cache_dir,
            filename=bin_file
        )
        logging.info(f"Successfully downloaded {bin_file}")
    except Exception as e:
        logging.error(f"Failed to download {bin_file}: {e}")
