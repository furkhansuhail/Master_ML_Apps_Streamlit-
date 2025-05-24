import sys
from pathlib import Path
import streamlit as st
import urllib.request as request
from urllib.parse import urlparse
# Add root directory to sys.path for module import
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import from your utility module
from ImportDataModules import download_file_from_github, DataIngestionConfig

# --- Set up Dataset folder ---
BASE_DIR = Path("Dataset")
BASE_DIR.mkdir(parents=True, exist_ok=True)

# --- Streamlit UI ---
st.title("📁 GitHub File Downloader (PDF / ZIP / CSV)")
github_url = st.text_input("🔗 Enter the GitHub Raw File URL:")


def convert_to_raw_url(github_url: str) -> str:
    """
    Convert a normal GitHub file URL to a raw content URL.

    Example:
    https://github.com/user/repo/blob/branch/path/to/file.pdf
    → https://raw.githubusercontent.com/user/repo/branch/path/to/file.pdf
    """
    if "github.com" in github_url and "/blob/" in github_url:
        raw_url = github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob", "")
        return raw_url
    else:
        raise ValueError("Invalid GitHub file URL format. Make sure it includes '/blob/'.")

def download_project_file(source_URL, local_data_file):
    local_data_file.parent.mkdir(parents=True, exist_ok=True)
    if local_data_file.exists():
        print(f"✅ File already exists at: {local_data_file}")
    else:
        print(f"⬇ Downloading file from {source_URL}...")
        file_path, _ = request.urlretrieve(url=source_URL, filename=local_data_file)
        print(f"✅ File downloaded and saved to: {file_path}")


def convert_to_raw_url(github_url: str) -> str:
    """
    Converts a GitHub file URL to a raw URL.
    If already raw, return as is.
    If it's a blob URL, convert to raw.githubusercontent.com format.
    """
    if "raw.githubusercontent.com" in github_url:
        return github_url

    if "github.com" in github_url and "/blob/" in github_url:
        return github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob", "")

    raise ValueError("Invalid GitHub URL format. Must be a raw URL or include '/blob/'.")


# --- Main action ---
if st.button("Download and Process"):

    if not github_url.strip():
        st.error("❌ Please enter a valid GitHub raw URL.")
    else:
        print(github_url)
        rawURL = convert_to_raw_url(github_url)

        filename = rawURL.split("/")[-1]

        local_path = BASE_DIR / filename
        local_data_file= Path(local_path)

        status_File = Path(BASE_DIR /"status.txt")
        print(github_url)
        print(rawURL)
        print("\n")

        print(local_path)
        print(local_path)
        print(status_File)
        print("\n")

        print(filename)


        # Setup config object (optional, useful for later LLM processing)
        config = DataIngestionConfig(
            root_dir=BASE_DIR,
            source_URL=github_url,
            local_data_file=local_data_file,
            STATUS_FILE= status_File,
            ALL_REQUIRED_FILES=[]
        )

        download_project_file(config.source_URL, config.local_data_file)
        print("_______________________________________________________________________________________________________")
