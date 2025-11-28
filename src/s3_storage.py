# src/s3_storage.py

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import boto3

from .config import get_settings


_settings = get_settings()


def get_s3_client():
    return boto3.client("s3", region_name=_settings.aws_region)


def upload_pdf_bytes(contents: bytes, filename: str) -> Tuple[str, str]:
    """
    Upload raw PDF bytes to S3.
    Returns (doc_id, s3_key).
    """
    # Basic doc_id from filename (without extension); you could use uuid4() later
    doc_id = Path(filename).stem

    s3_key = f"docs/{doc_id}/{filename}"

    client = get_s3_client()
    client.put_object(
        Bucket=_settings.s3_bucket,
        Key=s3_key,
        Body=contents,
        ContentType="application/pdf",
    )

    return doc_id, s3_key
