# src/config.py

from dataclasses import dataclass
import os

from dotenv import load_dotenv

# Load .env when this module is imported
load_dotenv()


@dataclass
class Settings:
    aws_region: str
    s3_bucket: str


def get_settings() -> Settings:
    region = os.getenv("AWS_REGION", "us-east-2")
    bucket = os.getenv("AWS_S3_BUCKET")
    if not bucket:
        raise RuntimeError("AWS_S3_BUCKET is not set in .env")
    return Settings(aws_region=region, s3_bucket=bucket)
