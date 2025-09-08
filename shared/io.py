# /shared/io.py
"""
Unified IO: Local Parquet + Azure Blob storage IO.

Public API:
- read_parquet(path: str) -> DataFrame
- write_parquet(df: DataFrame, path: str)
- get_blob_client() -> BlobServiceClient
- read_blob_parquet(container: str, blob: str) -> DataFrame (optional)
- write_blob_parquet(df: DataFrame, container: str, blob: str) (optional)
"""

import os
from typing import Optional
import pandas as pd
import io
import tempfile

# Azure imports are loaded lazily to keep optional in non-Azure environments
try:
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient
except Exception as e:
    # We don't raise here to allow non-Azure environments to import the module.
    BlobServiceClient = None  # type: ignore
    DefaultAzureCredential = None  # type: ignore
    _AZURE_SDK_UNAVAILABLE = True
else:
    _AZURE_SDK_UNAVAILABLE = False


def get_blob_client():
    if _AZURE_SDK_UNAVAILABLE:
        raise RuntimeError("Azure SDK not available. Install 'azure-identity' and 'azure-storage-blob'.")

    account_url = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
    if not account_url:
        raise ValueError("AZURE_STORAGE_ACCOUNT_URL environment variable is not set.")

    try:
        credential = DefaultAzureCredential()
        return BlobServiceClient(account_url=account_url, credential=credential)
    except Exception as e:
        raise RuntimeError(f"Failed to create BlobServiceClient: {e}") from e

def read_parquet(path: str):
    # Implementation: read a local Parquet file
    import pandas as pd
    return pd.read_parquet(path)

def write_parquet(df, path: str):
    # Implementation: write a local Parquet file
    df.to_parquet(path, index=False)

def read_blob_parquet(account_url: str, container: str, blob_name: str):
    # Azure path is optional; if azure SDK is unavailable, raise a clear error
    try:
        from azure.storage.blob import BlobClient  # type: ignore
    except Exception as e:
        raise RuntimeError("Azure SDK not available. Install 'azure-storage-blob' to use read_blob_parquet.") from e
    blob = BlobClient(account_url, container, blob_name)
    stream = blob.download_blob()
    import pandas as pd
    import io
    buf = io.BytesIO(stream.readall())
    return pd.read_parquet(buf)

def write_blob_parquet(account_url: str, container: str, blob_name: str, df):
    try:
        from azure.storage.blob import BlobClient  # type: ignore
    except Exception as e:
        raise RuntimeError("Azure SDK not available. Install 'azure-storage-blob' to use write_blob_parquet.") from e
    blob = BlobClient(account_url, container, blob_name)
    import io
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    blob.upload_blob(buf, overwrite=True)