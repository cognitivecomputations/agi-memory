from __future__ import annotations

import os
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest

import core.auth.spotify as spotify
import core.auth.store as auth_store
from core.integration_reliability import IntegrationHttpResponse
from core.tools.base import ToolContext, ToolExecutionContext
from core.tools.life_integrations import RevokeLifeIntegrationHandler


pytestmark = [pytest.mark.asyncio(loop_scope="session")]


def _context(pool) -> ToolExecutionContext:
    return ToolExecutionContext(
        tool_context=ToolContext.CHAT,
        call_id="spotify-integration-test",
        session_id="spotify-integration-test",
        registry=SimpleNamespace(pool=pool),
    )


async def test_spotify_pkce_full_connection_api_call_and_revoke(db_pool, monkeypatch, tmp_path):
    monkeypatch.setattr(auth_store, "AUTH_DIR", tmp_path / "auth")
    monkeypatch.delenv("HEXIS_SPOTIFY_REDIRECT_URI", raising=False)
    monkeypatch.delenv("HEXIS_API_URL", raising=False)
    monkeypatch.delenv("HEXIS_API_BASE_URL", raising=False)

    started = await spotify.start_spotify_oauth(
        db_pool,
        capabilities=["search", "playback_state", "playback_control"],
        client_id="spotify-public-client-id",
        source_channel="test",
        source_session_id="spotify-integration-test",
    )
    assert isinstance(started, spotify.SpotifyOAuthStart)
    payload = started.attempt_payload
    parsed = urlparse(payload["authorization_url"])
    params = parse_qs(parsed.query)
    assert parsed.scheme == "https"
    assert params["client_id"] == ["spotify-public-client-id"]
    assert params["code_challenge_method"] == ["S256"]
    assert params["redirect_uri"] == [
        "http://127.0.0.1:43817/api/integrations/spotify/callback"
    ]
    assert params["scope"] == [
        "user-read-private user-read-playback-state user-modify-playback-state"
    ]
    assert "client_secret" not in params

    pending = auth_store.load_auth(started.pending_auth_ref)
    assert isinstance(pending, dict)
    assert pending["verifier"]
    assert pending["state"]

    async def fake_request(provider, method, url, **kwargs):
        if provider == "spotify_oauth":
            assert method == "POST"
            assert kwargs["data"]["client_id"] == "spotify-public-client-id"
            assert kwargs["data"]["code_verifier"] == pending["verifier"]
            return {
                "access_token": "spotify-access-token",
                "refresh_token": "spotify-refresh-token",
                "expires_in": 3600,
                "scope": "user-read-private user-read-playback-state user-modify-playback-state",
            }
        if provider == "spotify" and url.endswith("/me"):
            assert kwargs["headers"]["Authorization"] == "Bearer spotify-access-token"
            return {
                "id": "spotify-user-1",
                "display_name": "Spotify User",
                "product": "premium",
                "country": "US",
            }
        raise AssertionError(f"unexpected request {provider} {method} {url}")

    monkeypatch.setattr(spotify, "request_json", fake_request)
    completed = await spotify.complete_spotify_oauth(
        db_pool,
        attempt_id=payload["attempt_id"],
        authorization_response=(
            "http://127.0.0.1:43817/api/integrations/spotify/callback"
            f"?code=spotify-code&state={pending['state']}"
        ),
    )
    assert completed.account_key == "spotify:spotify-user-1"
    assert completed.display_name == "Spotify User"
    assert completed.capabilities == ["search", "playback_state", "playback_control"]
    assert auth_store.load_auth(started.pending_auth_ref) is None

    credentials = auth_store.load_auth(spotify.SPOTIFY_DEFAULT_CREDENTIAL_REF)
    assert credentials["access_token"] == "spotify-access-token"
    auth_file = auth_store.AUTH_DIR / f"{spotify.SPOTIFY_DEFAULT_CREDENTIAL_REF}.json"
    assert auth_file.exists()
    assert os.stat(auth_file).st_mode & 0o777 == 0o600

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT credential_ref, metadata::text AS metadata_text, status
            FROM integration_connections
            WHERE connector_id = 'spotify' AND account_key = $1
            """,
            completed.account_key,
        )
    assert row["credential_ref"] == spotify.SPOTIFY_DEFAULT_CREDENTIAL_REF
    assert row["status"] == "connected"
    assert "spotify-access-token" not in row["metadata_text"]
    assert "spotify-refresh-token" not in row["metadata_text"]

    async def fake_response(provider, method, url, **kwargs):
        assert provider == "spotify"
        assert method == "GET"
        assert url.endswith("/me/player")
        assert kwargs["headers"]["Authorization"] == "Bearer spotify-access-token"
        return IntegrationHttpResponse(
            status_code=200,
            headers={},
            text='{"is_playing":true}',
            json_data={"is_playing": True},
            correlation_id="spotify-test",
        )

    monkeypatch.setattr(spotify, "request_json_response", fake_response)
    playback = await spotify.spotify_api_request(
        db_pool,
        "GET",
        "/me/player",
        capability="playback_state",
        account_key=completed.account_key,
    )
    assert playback == {"is_playing": True}

    revoked = await RevokeLifeIntegrationHandler().execute(
        {
            "connector_id": "spotify",
            "account_key": completed.account_key,
            "reason": "test complete",
        },
        _context(db_pool),
    )
    assert revoked.success is True
    assert revoked.output["revoked"] == 1
    assert auth_store.load_auth(spotify.SPOTIFY_DEFAULT_CREDENTIAL_REF) is None


async def test_spotify_requires_explicit_env_choice_and_loopback_ip(db_pool, monkeypatch, tmp_path):
    monkeypatch.setattr(auth_store, "AUTH_DIR", tmp_path / "auth-2")
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "ambient-client-id")
    monkeypatch.delenv("HEXIS_SPOTIFY_REDIRECT_URI", raising=False)

    started = await spotify.start_spotify_oauth(db_pool)
    assert isinstance(started, dict)
    assert started["status"] == "needs_client"
    assert "ambient-client-id" not in str(started)

    monkeypatch.setenv(
        "HEXIS_SPOTIFY_REDIRECT_URI",
        "http://localhost:43817/api/integrations/spotify/callback",
    )
    with pytest.raises(spotify.SpotifyOAuthError, match="loopback IP address"):
        spotify.configured_spotify_redirect_uri()

    monkeypatch.setenv(
        "HEXIS_SPOTIFY_REDIRECT_URI",
        "http://example.com/api/integrations/spotify/callback",
    )
    with pytest.raises(spotify.SpotifyOAuthError, match="loopback IP address"):
        spotify.configured_spotify_redirect_uri()

    monkeypatch.setenv(
        "HEXIS_SPOTIFY_REDIRECT_URI",
        "https://hexis.example/api/integrations/spotify/callback",
    )
    assert spotify.configured_spotify_redirect_uri() == (
        "https://hexis.example/api/integrations/spotify/callback"
    )
