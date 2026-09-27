"""Bounded, allowlisted public JSON fetching with immutable raw snapshots."""

from __future__ import annotations

import hashlib
import json
import math
import ssl
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener


DEFAULT_HOSTS = frozenset({"statsapi.mlb.com", "earthquake.usgs.gov", "gamma-api.polymarket.com", "clob.polymarket.com"})


class FetchError(RuntimeError):
    pass


def strict_json_loads(text):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError("Non-finite JSON number")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Non-finite JSON number")
        return number

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant, parse_float=finite_float)


class _AllowedRedirect(HTTPRedirectHandler):
    def __init__(self, check):
        self.check = check

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.check(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class HttpClient:
    def __init__(self, snapshot_dir="runs/snapshots", *, timeout=15, max_bytes=8_000_000,
                 attempts=3, allowed_hosts=DEFAULT_HOSTS, clock=None, opener=None, sleep=time.sleep):
        if not 0 < timeout <= 60 or not 1 <= attempts <= 5 or not 0 < max_bytes <= 32_000_000:
            raise ValueError("Invalid HTTP limits")
        self.snapshot_dir = Path(snapshot_dir)
        self.timeout, self.max_bytes, self.attempts = timeout, max_bytes, attempts
        self.allowed_hosts = frozenset(allowed_hosts)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        context = ssl.create_default_context()
        system_ca = Path("/etc/ssl/cert.pem")
        if system_ca.is_file():
            context.load_verify_locations(cafile=str(system_ca))
        self.opener = opener or build_opener(_AllowedRedirect(self._check_url), HTTPSHandler(context=context))
        self.sleep = sleep
        self.last_snapshot = None
        self.snapshots = []

    def _check_url(self, url):
        parsed = urlsplit(url)
        if (parsed.scheme != "https" or parsed.hostname not in self.allowed_hosts or parsed.username
                or parsed.password or parsed.port not in {None, 443} or "\\" in url):
            raise FetchError("Collector URL is outside the HTTPS source allowlist")
        return parsed.hostname

    def get_json(self, url):
        self.last_snapshot = None
        host = self._check_url(url)
        for attempt in range(self.attempts):
            try:
                request = Request(url, headers={"User-Agent": "FuturePredictionBench/0.2 (public research data collection)",
                                                "Accept": "application/json", "Accept-Encoding": "identity"})
                with self.opener.open(request, timeout=self.timeout) as response:
                    final_url = response.geturl()
                    self._check_url(final_url)
                    raw = response.read(self.max_bytes + 1)
                    if len(raw) > self.max_bytes:
                        raise FetchError("Source response exceeded the byte limit")
                payload = strict_json_loads(raw.decode("utf-8"))
                now = self.clock()
                if now.tzinfo is None:
                    raise ValueError("Snapshot clock must have a timezone")
                digest = hashlib.sha256(raw).hexdigest()
                self.snapshot_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
                path = self.snapshot_dir / (digest + ".json")
                if not path.exists():
                    with path.open("xb") as handle:
                        handle.write(raw)
                    path.chmod(0o600)
                snapshot = {"url": final_url, "requested_url": url,
                            "observed_at": now.astimezone(timezone.utc).isoformat(),
                            "sha256": digest, "path": str(path), "bytes": len(raw)}
                self.last_snapshot = snapshot
                self.snapshots.append(snapshot)
                return payload
            except HTTPError as exc:
                if exc.code not in {408, 429, 500, 502, 503, 504} or attempt + 1 == self.attempts:
                    raise FetchError(f"HTTP {exc.code} from {host}") from None
            except (URLError, TimeoutError, OSError) as exc:
                if attempt + 1 == self.attempts:
                    raise FetchError(f"{type(exc).__name__} fetching {host}") from None
            except (ValueError, UnicodeError) as exc:
                raise FetchError(f"Invalid JSON response from {host}: {type(exc).__name__}") from None
            self.sleep(min(2 ** attempt, 4))
        raise FetchError("Source request failed")
