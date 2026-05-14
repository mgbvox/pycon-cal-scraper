"""OAuth desktop flow for Google Calendar.

The user supplies a ``client_secret.json`` downloaded from Google Cloud
Console (OAuth 2.0 Client ID, type "Desktop"). After the first
:func:`login` call, the resulting credentials are cached at
:func:`pycon_cal_scraper.paths.token_file` and reused (with silent refresh)
on subsequent runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from pycon_cal_scraper.paths import token_file

SCOPES: tuple[str, ...] = ("https://www.googleapis.com/auth/calendar.events",)


class _FlowFactory(Protocol):
    """Factory protocol for the OAuth installed-app flow.

    Exists purely so tests can inject a stub instead of running a real
    browser-based OAuth dance.
    """

    def __call__(self, client_secret_path: Path, scopes: tuple[str, ...]) -> Any: ...


def _default_flow_factory(client_secret_path: Path, scopes: tuple[str, ...]) -> Any:
    """Build a real :class:`InstalledAppFlow` from a ``client_secret.json`` file."""
    return InstalledAppFlow.from_client_secrets_file(str(client_secret_path), list(scopes))


def load_cached_credentials(token_path: Path | None = None) -> Credentials | None:
    """Load cached OAuth credentials from disk, refreshing if expired.

    Args:
        token_path: Where to read the token from. Defaults to
            :func:`pycon_cal_scraper.paths.token_file`.

    Returns:
        The credentials, or ``None`` if no token file exists. If the cached
        token is expired but has a refresh token, this function refreshes
        it and rewrites the file on disk before returning.
    """
    path = token_path or token_file()
    if not path.exists():
        return None
    creds = Credentials.from_authorized_user_file(str(path), list(SCOPES))
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        path.write_text(creds.to_json(), encoding="utf-8")
    return creds


def login(
    client_secret_path: Path,
    *,
    token_path: Path | None = None,
    flow_factory: _FlowFactory = _default_flow_factory,
) -> Credentials:
    """Return valid OAuth credentials, running the browser flow if needed.

    Reuses any cached, valid token first. If none exists, runs the OAuth
    desktop flow (opens a browser, listens on a local port), then caches
    the resulting credentials to ``token_path`` for future runs.

    Args:
        client_secret_path: Path to a Desktop OAuth client JSON downloaded
            from Google Cloud Console.
        token_path: Where to read/write the cached token. Defaults to
            :func:`pycon_cal_scraper.paths.token_file`.
        flow_factory: Hook used to construct the OAuth flow; tests inject
            a fake. Defaults to :func:`_default_flow_factory`.

    Returns:
        Valid :class:`Credentials` ready to pass to
        :func:`build_calendar_service`.
    """
    path = token_path or token_file()
    creds = load_cached_credentials(path)
    if creds and creds.valid:
        return creds
    flow = flow_factory(client_secret_path, SCOPES)
    creds = flow.run_local_server(port=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(creds.to_json(), encoding="utf-8")
    return creds


def build_calendar_service(credentials: Credentials) -> Any:
    """Build a Google Calendar v3 service client.

    Args:
        credentials: OAuth credentials from :func:`login`.

    Returns:
        A ``googleapiclient`` discovery service. Discovery doc caching is
        disabled to keep import-time side effects out of test runs.
    """
    return build("calendar", "v3", credentials=credentials, cache_discovery=False)
