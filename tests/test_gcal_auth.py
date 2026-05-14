"""Tests for the gcal OAuth flow (token reuse path; live flow is not tested)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pycon_cal_scraper.gcal import auth


class _FakeCreds:
    def __init__(self, valid: bool = True) -> None:
        self.valid = valid
        self.expired = False
        self.refresh_token = None

    def to_json(self) -> str:
        return json.dumps({"token": "abc", "refresh_token": "r", "client_id": "c"})


def test_login_uses_cached_credentials_when_valid(tmp_path: Path, monkeypatch: object) -> None:
    token_path = tmp_path / "token.json"
    token_path.write_text(_FakeCreds().to_json(), encoding="utf-8")

    fake = _FakeCreds(valid=True)
    monkeypatch.setattr(  # type: ignore[attr-defined]
        auth.Credentials,
        "from_authorized_user_file",
        staticmethod(lambda *a, **k: fake),
    )

    def boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("flow_factory must not be invoked when cached creds are valid")

    result = auth.login(tmp_path / "client_secret.json", token_path=token_path, flow_factory=boom)
    assert result is fake


def test_login_runs_flow_when_no_cached_creds(tmp_path: Path, monkeypatch: object) -> None:
    token_path = tmp_path / "token.json"

    class FakeFlow:
        def run_local_server(self, port: int = 0) -> _FakeCreds:
            return _FakeCreds(valid=True)

    def factory(client_secret_path: Path, scopes: tuple[str, ...]) -> FakeFlow:
        return FakeFlow()

    result = auth.login(
        tmp_path / "client_secret.json", token_path=token_path, flow_factory=factory
    )
    assert isinstance(result, _FakeCreds)
    assert token_path.exists()
