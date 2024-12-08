from huggingface_hub import snapshot_download
import os
import shutil

# Specify the model repository and the destination directory
repo_id = "EleutherAI/pythia-160m"
output_dir = "pythia-160m"

# Download the model files from the Hugging Face Hub
print("Downloading Pythia 160M...")
cache_dir = snapshot_download(repo_id=repo_id)

# Move files to the desired output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for file_name in [
    "README.md",
    "config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json"
]:
    src_file = os.path.join(cache_dir, file_name)
    dst_file = os.path.join(output_dir, file_name)
    if os.path.exists(src_file):
        shutil.copy(src_file, dst_file)
    else:
        print(f"Warning: {file_name} not found in the repository!")

print(f"Pythia 160M files are saved in the '{output_dir}' folder.")
