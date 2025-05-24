from pathlib import Path
from dataclasses import dataclass
import urllib.request as request
import zipfile
import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    STATUS_FILE: Path
    ALL_REQUIRED_FILES: list




def download_project_file(source_URL, local_data_file):
    local_data_file.parent.mkdir(parents=True, exist_ok=True)
    if local_data_file.exists():
        print(f"✅ File already exists at: {local_data_file}")
    else:
        print(f"⬇ Downloading file from {source_URL}...")
        file_path, _ = request.urlretrieve(url=source_URL, filename=local_data_file)
        print(f"✅ File downloaded and saved to: {file_path}")


def download_file_from_github(github_url: str, save_dir: str = "Dataset", streamlit_ui=None):
    st = streamlit_ui if streamlit_ui else None

    # Convert to raw GitHub URL if needed
    if "github.com" in github_url and "/blob/" in github_url:
        github_url = github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob", "")

    save_path = Path(save_dir) / github_url.split("/")[-1]
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip download if file already exists
    if save_path.exists():
        msg = f"✅ File already exists at: `{save_path}`. Skipping download."
        if st:
            st.info(msg)
        else:
            print(msg)
        return save_path

    try:
        if st: st.info(f"⬇ Downloading from: {github_url}")
        request.urlretrieve(github_url, save_path)
        if st: st.success(f"✅ File saved to: `{save_path}`")
        else: print(f"✅ File saved to: {save_path}")

        # Handle file types
        if save_path.suffix.lower() == ".zip":
            extract_path = Path(save_dir) / save_path.stem
            with zipfile.ZipFile(save_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            extracted_files = list(extract_path.glob("**/*"))

            if st:
                st.info(f"📦 ZIP Extracted to `{extract_path}` with {len(extracted_files)} files:")
                for f in extracted_files:
                    st.write(f"- {f}")
                    if f.suffix.lower() == ".csv":
                        preview_csv(f, st)

        elif save_path.suffix.lower() == ".csv":
            preview_csv(save_path, st)

        elif save_path.suffix.lower() == ".pdf":
            if st: st.info("📄 PDF file downloaded. Ready for further processing.")
            else: print("📄 PDF downloaded.")

        else:
            if st: st.warning("⚠️ Unknown file type. No specific preview available.")

        return save_path

    except Exception as e:
        if st: st.error(f"❌ Failed to download/process file: {e}")
        else: print(f"❌ Error: {e}")
        raise e




def preview_csv(file_path: Path, st):
    try:
        df = pd.read_csv(file_path)
        st.write(f"📊 Preview of `{file_path.name}`:")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")
