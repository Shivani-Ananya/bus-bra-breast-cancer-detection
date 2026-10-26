"""Minimal authenticated Supabase REST client for the Vercel API.

Requests use the public project key plus the caller's Supabase access token.
PostgreSQL Row-Level Security remains the authorization boundary; the service
role key is never sent to a browser or used for ordinary case access.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from fastapi import HTTPException


@dataclass(frozen=True)
class SupabaseSettings:
    url: str
    anon_key: str
    service_role_key: str

    @property
    def configured(self) -> bool:
        return bool(self.url and self.anon_key)


def settings() -> SupabaseSettings:
    return SupabaseSettings(
        url=os.getenv("SUPABASE_URL", "").rstrip("/"),
        anon_key=os.getenv("SUPABASE_ANON_KEY", ""),
        service_role_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
    )


def bearer_token(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="A Supabase access token is required.")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="A Supabase access token is required.")
    return token


def user_request(method: str, resource: str, token: str, *,
                 query: dict[str, str] | None = None,
                 payload: dict[str, Any] | None = None) -> Any:
    cfg = settings()
    if not cfg.configured:
        raise HTTPException(status_code=503, detail="Supabase is not configured.")
    url = f"{cfg.url}/rest/v1/{resource.lstrip('/')}"
    if query:
        url += "?" + urlencode(query, safe=",().*:")
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(url, data=body, method=method.upper(), headers={
        "apikey": cfg.anon_key,
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Prefer": "return=representation",
    })
    try:
        with urlopen(request, timeout=15) as response:
            content = response.read()
            return json.loads(content) if content else None
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(detail).get("message", detail)
        except json.JSONDecodeError:
            pass
        raise HTTPException(status_code=exc.code, detail=detail) from exc
    except (URLError, TimeoutError) as exc:
        raise HTTPException(status_code=502, detail="Supabase is temporarily unreachable.") from exc


def auth_user(token: str) -> dict[str, Any]:
    """Return the authenticated Supabase user for a caller token."""
    cfg = settings()
    if not cfg.configured:
        raise HTTPException(status_code=503, detail="Supabase is not configured.")
    request = Request(f"{cfg.url}/auth/v1/user", headers={
        "apikey": cfg.anon_key,
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    })
    try:
        with urlopen(request, timeout=15) as response:
            return json.loads(response.read())
    except HTTPError as exc:
        raise HTTPException(status_code=401, detail="The Supabase session is invalid or expired.") from exc
    except (URLError, TimeoutError) as exc:
        raise HTTPException(status_code=502, detail="Supabase is temporarily unreachable.") from exc


def storage_signed_url(path: str, token: str, expires_in: int = 300) -> str:
    """Create a short-lived URL for an object the signed-in user may read."""
    cfg = settings()
    body = json.dumps({"expiresIn": expires_in}).encode("utf-8")
    request = Request(
        f"{cfg.url}/storage/v1/object/sign/breastsight-private/{path}",
        data=body,
        method="POST",
        headers={
            "apikey": cfg.anon_key,
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=15) as response:
            result = json.loads(response.read())
            signed = result.get("signedURL") or result.get("signedUrl")
            if not signed:
                raise HTTPException(status_code=502, detail="Supabase did not return an image URL.")
            return signed if signed.startswith("http") else f"{cfg.url}/storage/v1{signed}"
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=exc.code, detail=detail) from exc
    except (URLError, TimeoutError) as exc:
        raise HTTPException(status_code=502, detail="Supabase Storage is temporarily unreachable.") from exc


def database_probe() -> bool:
    """Confirm the configured service can read the migrated schema.

    The probe returns only a boolean and never returns credentials or rows.
    """
    cfg = settings()
    if not cfg.configured or not cfg.service_role_key:
        return False
    request = Request(
        f"{cfg.url}/rest/v1/cases?select=id&limit=1",
        headers={
            "apikey": cfg.service_role_key,
            "Authorization": f"Bearer {cfg.service_role_key}",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=10) as response:
            response.read()
            return 200 <= response.status < 300
    except (HTTPError, URLError, TimeoutError):
        return False
