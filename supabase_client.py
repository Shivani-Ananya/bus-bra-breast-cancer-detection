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
