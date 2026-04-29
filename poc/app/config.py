from dataclasses import dataclass
import os


@dataclass(frozen=True)
class AppConfig:
    runpod_api_key: str
    runpod_endpoint_id: str
    r2_account_id: str
    r2_access_key_id: str
    r2_secret_access_key: str
    r2_bucket: str
    r2_public_base_url: str | None = None
    data_dir: str = "poc/.data"

    @classmethod
    def required_env_names(cls) -> tuple[str, ...]:
        return (
            "RUNPOD_API_KEY",
            "RUNPOD_ENDPOINT_ID",
            "R2_ACCOUNT_ID",
            "R2_ACCESS_KEY_ID",
            "R2_SECRET_ACCESS_KEY",
            "R2_BUCKET",
        )

    @property
    def r2_endpoint_url(self) -> str:
        return f"https://{self.r2_account_id}.r2.cloudflarestorage.com"


def require_env() -> AppConfig:
    missing = [name for name in AppConfig.required_env_names() if not os.getenv(name)]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variables: {joined}")

    public_base = os.getenv("R2_PUBLIC_BASE_URL")
    return AppConfig(
        runpod_api_key=os.environ["RUNPOD_API_KEY"],
        runpod_endpoint_id=os.environ["RUNPOD_ENDPOINT_ID"],
        r2_account_id=os.environ["R2_ACCOUNT_ID"],
        r2_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        r2_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        r2_bucket=os.environ["R2_BUCKET"],
        r2_public_base_url=public_base.rstrip("/") if public_base else None,
        data_dir=os.getenv("POC_DATA_DIR", "poc/.data"),
    )
