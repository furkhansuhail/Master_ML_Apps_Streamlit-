# THis is my Import DataModules
# from ImportsForRag import *
from pathlib import Path
from dataclasses import dataclass
import urllib.request as request


import streamlit as st
import os
import zipfile
import pandas as pd

from pathlib import Path
from urllib import request
import zipfile
import pandas as pd

def download_file_from_github(github_url: str, save_dir: str = "Dataset", streamlit_ui=None):
    st = streamlit_ui if streamlit_ui else None
    save_path = Path(save_dir) / github_url.split("/")[-1]
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if st: st.info(f"⬇ Downloading from: {github_url}")
        request.urlretrieve(github_url, save_path)
        if st: st.success(f"✅ File saved to: {save_path}")
        else: print(f"✅ File saved to: {save_path}")

        # Handle file types
        if save_path.suffix.lower() == ".zip":
            extract_path = Path(save_dir) / save_path.stem
            with zipfile.ZipFile(save_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            extracted_files = list(extract_path.glob("**/*"))

            if st:
                st.info(f"📦 ZIP Extracted to {extract_path} with {len(extracted_files)} files:")
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
        st.write(f"📊 Preview of {file_path.name}:")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")



# @dataclass(frozen=True)
# class DataIngestionConfig:
#     root_dir: Path
#     source_URL: str
#     local_data_file: Path
#     STATUS_FILE: Path
#     ALL_REQUIRED_FILES: list
#
#
# def download_file_from_github(github_url: str, save_dir: str = "Dataset", streamlit_ui=None):
#     """
#     Downloads a file from GitHub raw URL and processes it (extract ZIP, preview CSV).
#     Args:
#         github_url (str): GitHub raw URL to download
#         save_dir (str): Local directory to save file
#         streamlit_ui (module): Optional, Streamlit module for UI output (e.g. st)
#     Returns:
#         Path: Path to the downloaded file
#     """
#     st = streamlit_ui if streamlit_ui else None
#     save_path = Path(save_dir) / github_url.split("/")[-1]
#     save_path.parent.mkdir(parents=True, exist_ok=True)
#
#     try:
#         if st: st.info(f"⬇ Downloading from: {github_url}")
#         request.urlretrieve(github_url, save_path)
#         if st: st.success(f"✅ File saved to: {save_path}")
#         else: print(f"✅ File saved to: {save_path}")
#
#         # Handle file types
#         if save_path.suffix.lower() == ".zip":
#             extract_path = Path(save_dir) / save_path.stem
#             with zipfile.ZipFile(save_path, 'r') as zip_ref:
#                 zip_ref.extractall(extract_path)
#             extracted_files = list(extract_path.glob("**/*"))
#
#             if st:
#                 st.info(f"📦 ZIP Extracted to {extract_path} with {len(extracted_files)} files:")
#                 for f in extracted_files:
#                     st.write(f"- {f}")
#                     if f.suffix.lower() == ".csv":
#                         preview_csv(f, st)
#             else:
#                 print(f"📦 ZIP extracted to: {extract_path}")
#
#         elif save_path.suffix.lower() == ".csv":
#             preview_csv(save_path, st)
#
#         elif save_path.suffix.lower() == ".pdf":
#             if st: st.info("📄 PDF file downloaded. Ready for further processing.")
#             else: print("📄 PDF downloaded.")
#
#         else:
#             if st: st.warning("⚠️ Unknown file type. No specific preview available.")
#
#         return save_path
#
#     except Exception as e:
#         if st: st.error(f"❌ Failed to download/process file: {e}")
#         else: print(f"❌ Error: {e}")
#         raise e
#
#
# def preview_csv(file_path: Path, st):
#     try:
#         df = pd.read_csv(file_path)
#         st.write(f"📊 Preview of {file_path.name}:")
#         st.dataframe(df.head())
#     except Exception as e:
#         st.error(f"❌ Error reading CSV: {e}")

# THis is my streamit main function

import sys
from pathlib import Path
from ImportDataModules import *

# Add root directory to sys.path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import from your shared module
from ImportDataModules import download_file_from_github

# --- Set up base Dataset directory ---
BASE_DIR = Path("Dataset")
BASE_DIR.mkdir(parents=True, exist_ok=True)

# --- Streamlit UI ---
st.title("📁 GitHub File Downloader (PDF / ZIP / CSV)")
github_url = st.text_input("🔗 Enter the GitHub Raw File URL:")

# --- Main Action ---
if st.button("Download and Process"):
    if not github_url.strip():
        st.error("❌ Please enter a valid GitHub raw URL.")
    else:
        filename = github_url.split("/")[-1]
        local_path = BASE_DIR / filename

        # Setup config object
        config = DataIngestionConfig(
            root_dir=BASE_DIR,
            source_URL=github_url,
            local_data_file=local_path,
            STATUS_FILE=BASE_DIR / "status.txt",
            ALL_REQUIRED_FILES=[]
        )

        # Use shared download + processing function
        download_file_from_github(github_url, save_dir="Dataset", streamlit_ui=st)

# from ImportDataModules import *

# # --- Main action ---
# if st.button("Download and Process"):
#     if not github_url.strip():
#         st.error("❌ Please enter a valid GitHub raw URL.")
#     else:
#         filename = github_url.split("/")[-1]
#         local_path = BASE_DIR / filename
#
#         config = DataIngestionConfig(
#             root_dir=BASE_DIR,
#             source_URL=github_url,
#             local_data_file=local_path,
#             STATUS_FILE=BASE_DIR / "status.txt",
#             ALL_REQUIRED_FILES=[]
#         )
#
#         try:
#             file_path = download_file(config.source_URL, config.local_data_file)
#             st.success(f"✅ File downloaded to: {file_path}")
#
#             if file_path.suffix.lower() == ".zip":
#                 extract_path = BASE_DIR / file_path.stem
#                 extracted_files = extract_zip(file_path, extract_path)
#                 st.info(f"📦 ZIP Extracted to {extract_path} with {len(extracted_files)} files:")
#                 for f in extracted_files:
#                     st.write(f"- {f}")
#                     if f.suffix.lower() == ".csv":
#                         preview_csv(f)
#
#             elif file_path.suffix.lower() == ".csv":
#                 preview_csv(file_path)
#
#             elif file_path.suffix.lower() == ".pdf":
#                 st.info("📄 PDF file downloaded. Ready for further processing.")
#
#             else:
#                 st.warning("⚠️ Unknown file type. No specific preview available.")
#
#         except Exception as e:
#             st.error(f"❌ Failed to download or process file: {e}")
#

#
# # Config class for ingestion
# @dataclass(frozen=True)
# class DataIngestionConfig:
#     root_dir: Path
#     source_URL: str
#     local_data_file: Path
#     STATUS_FILE: str
#     ALL_REQUIRED_FILES: list
#
# config = DataIngestionConfig(
#     root_dir=Path("Dataset"),
#     source_URL="https://raw.githubusercontent.com/furkhansuhail/ProjectData/main/RagModel_Bedrock_Data/human-nutrition-text.pdf",
#     local_data_file=Path("Dataset/human-nutrition-text.pdf"),
#     STATUS_FILE="Dataset/status.txt",
#     ALL_REQUIRED_FILES=[]
# )
#
# def download_project_file(source_URL, local_data_file):
#     local_data_file.parent.mkdir(parents=True, exist_ok=True)
#     if local_data_file.exists():
#         print(f"✅ File already exists at: {local_data_file}")
#     else:
#         print(f"⬇ Downloading file from {source_URL}...")
#         file_path, _ = request.urlretrieve(url=source_URL, filename=local_data_file)
#         print(f"✅ File downloaded and saved to: {file_path}")