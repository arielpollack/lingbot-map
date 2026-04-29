import pytest

from poc.app.config import AppConfig, require_env


def test_require_env_reads_required_values(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "endpoint-id")
    monkeypatch.setenv("R2_ACCOUNT_ID", "account")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "access")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("R2_BUCKET", "bucket")
    monkeypatch.setenv("R2_PUBLIC_BASE_URL", "https://cdn.example.com/base/")

    config = require_env()

    assert config.runpod_api_key == "rp-key"
    assert config.runpod_endpoint_id == "endpoint-id"
    assert config.r2_endpoint_url == "https://account.r2.cloudflarestorage.com"
    assert config.r2_public_base_url == "https://cdn.example.com/base"


def test_require_env_reports_missing_values(monkeypatch):
    for name in AppConfig.required_env_names():
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError) as exc:
        require_env()

    message = str(exc.value)
    assert "Missing required environment variables" in message
    assert "RUNPOD_API_KEY" in message
    assert "R2_BUCKET" in message
