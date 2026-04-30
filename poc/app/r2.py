from __future__ import annotations

from pathlib import Path
from typing import BinaryIO
from urllib.parse import quote

import boto3

from poc.app.config import AppConfig


class R2Client:
    def __init__(self, config: AppConfig):
        self.bucket = config.r2_bucket
        self.public_base_url = config.r2_public_base_url
        self.client = boto3.client(
            "s3",
            endpoint_url=config.r2_endpoint_url,
            aws_access_key_id=config.r2_access_key_id,
            aws_secret_access_key=config.r2_secret_access_key,
            region_name="auto",
        )

    def upload_fileobj(
        self,
        fileobj: BinaryIO,
        key: str,
        content_type: str | None = None,
        content_encoding: str | None = None,
    ) -> None:
        extra_args = _extra_args(content_type, content_encoding)
        if extra_args:
            self.client.upload_fileobj(fileobj, self.bucket, key, ExtraArgs=extra_args)
        else:
            self.client.upload_fileobj(fileobj, self.bucket, key)

    def upload_file(
        self,
        path: str | Path,
        key: str,
        content_type: str | None = None,
        content_encoding: str | None = None,
    ) -> None:
        extra_args = _extra_args(content_type, content_encoding)
        if extra_args:
            self.client.upload_file(str(path), self.bucket, key, ExtraArgs=extra_args)
        else:
            self.client.upload_file(str(path), self.bucket, key)

    def head_object(self, key: str) -> dict:
        return self.client.head_object(Bucket=self.bucket, Key=key)

    def download_file(self, key: str, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(path))

    def get_object_body(self, key: str):
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"]

    def public_url(self, key: str) -> str | None:
        if not self.public_base_url:
            return None
        return f"{self.public_base_url}/{quote(key)}"


def _extra_args(content_type: str | None, content_encoding: str | None) -> dict | None:
    extra: dict = {}
    if content_type:
        extra["ContentType"] = content_type
    if content_encoding:
        extra["ContentEncoding"] = content_encoding
    return extra or None
