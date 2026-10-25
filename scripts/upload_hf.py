"""Upload the trained weights to a Hugging Face model repo.

hf auth login                      # once, with a write token
python scripts/upload_hf.py USER/REPO --files student.pt hp.pt m1.pt [--github-repo OWNER/NAME]
Creates the repo (public) if needed and uploads the files and hf_model_card.md as README.md.
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("repo")
ap.add_argument("--files", nargs="+", required=True)
ap.add_argument("--github-repo", default="GITHUB_REPO")
a = ap.parse_args()
api = HfApi()  # token from `hf auth login` or the HF_TOKEN environment variable
api.create_repo(a.repo, repo_type="model", private=False, exist_ok=True)
card = Path(__file__).with_name("hf_model_card.md").read_text().replace("USER/REPO", a.repo)
card = card.replace("GITHUB_REPO", a.github_repo)
api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=a.repo)
for f in a.files:
    api.upload_file(path_or_fileobj=f, path_in_repo=Path(f).name, repo_id=a.repo)
    print("uploaded", f, flush=True)
print(f"https://huggingface.co/{a.repo}")
