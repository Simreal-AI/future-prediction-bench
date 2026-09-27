"""Optional live providers. Credentials never become model observations.

The default transport has no proxy/cookie support and does not execute pages.
Model endpoints are operator configuration; model-selected URLs are public-only.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import math
import os
import queue
import socket
import ssl
import threading
import time
from html.parser import HTMLParser
from urllib.parse import quote, urlencode, urljoin, urlsplit, urlunsplit


class ProviderError(RuntimeError):
    """A deliberately redacted transport or response error."""


def positive_int(value, name, maximum=1_000_000):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}")
    return value


def positive_seconds(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 3600:
        raise ValueError(f"{name} must be finite and between 0 and 3600 seconds")
    return float(value)


def strict_json_loads(value):
    """Reject ambiguous keys and NaN/Infinity, including overflow to infinity."""
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = item
        return result

    def number(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Non-finite JSON number")
        return result

    def invalid(value):
        raise ValueError("Non-finite JSON constant")

    return json.loads(value, object_pairs_hook=pairs, parse_float=number, parse_constant=invalid)


def _url_parts(url, public_only):
    if not isinstance(url, str) or not url or len(url) > 8192 or "\\" in url or any(c.isspace() or ord(c) < 32 for c in url):
        raise ProviderError("Invalid HTTP URL")
    try:
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username is not None or parts.password is not None:
            raise ValueError()
        if "%" in parts.hostname:
            raise ValueError()
        hostname = parts.hostname.rstrip(".").encode("idna").decode("ascii").lower()
        port = parts.port or (443 if parts.scheme == "https" else 80)
        if public_only and port not in {80, 443}:
            raise ValueError()
    except (UnicodeError, ValueError) as exc:
        raise ProviderError("Invalid HTTP URL") from None
    return parts, hostname, port


def _public_address(value):
    address = ipaddress.ip_address(value)
    if (not address.is_global or address.is_multicast or address.is_reserved or address.is_loopback
            or address.is_link_local or address.is_unspecified):
        return False
    if isinstance(address, ipaddress.IPv6Address):
        # Translation/tunnel ranges can encode a private destination.
        if address.ipv4_mapped or address.sixtofour or address.teredo:
            return False
        if address in ipaddress.ip_network("64:ff9b::/96") or address in ipaddress.ip_network("64:ff9b:1::/48"):
            return False
    return True


_DNS_SLOTS = threading.BoundedSemaphore(8)


def _resolve(hostname, port, timeout, public_only):
    """Resolve once, reject mixed public/private answers and pin the chosen IP.

    A timed-out OS resolver runs in a bounded daemon thread, never a fresh request.
    """
    if not _DNS_SLOTS.acquire(blocking=False):
        raise ProviderError("DNS concurrency limit reached")
    result = queue.Queue(maxsize=1)

    def lookup():
        try:
            result.put((True, socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)))
        except Exception:
            result.put((False, None))
        finally:
            _DNS_SLOTS.release()

    threading.Thread(target=lookup, daemon=True).start()
    try:
        ok, addresses = result.get(timeout=timeout)
    except queue.Empty:
        raise ProviderError("DNS timeout") from None
    if not ok or not addresses:
        raise ProviderError("DNS lookup failed")
    if public_only and any(not _public_address(item[4][0]) for item in addresses):
        raise ProviderError("Non-public destination blocked")
    return addresses[0]


def _request_bytes(url, *, method="GET", body=None, headers=None, timeout=30.0,
                   max_bytes=2_000_000, public_only=True):
    """One HTTP request, pinned DNS, TLS host verification and absolute timeout.

    Redirects are returned to the caller and never followed with credentials.
    """
    timeout = positive_seconds(timeout, "timeout")
    positive_int(max_bytes, "max_bytes", 10_000_000)
    started = time.monotonic()
    parts, hostname, port = _url_parts(url, public_only)
    address = _resolve(hostname, port, timeout, public_only)
    remaining = timeout - (time.monotonic() - started)
    if remaining <= 0:
        raise ProviderError("Request timeout")
    live_socket = [None]

    def abort():
        current = live_socket[0]
        if current is not None:
            try:
                current.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                current.close()
            except OSError:
                pass

    timer = threading.Timer(remaining, abort)
    timer.daemon = True
    connection = None
    try:
        family, kind, proto, _, sockaddr = address
        current = socket.socket(family, kind, proto)
        live_socket[0] = current
        timer.start()
        current.settimeout(remaining)
        current.connect(sockaddr)
        if parts.scheme == "https":
            context = ssl.create_default_context()
            current = context.wrap_socket(current, server_hostname=hostname, do_handshake_on_connect=False)
            live_socket[0] = current
            current.do_handshake()
        connection = http.client.HTTPConnection(hostname, port, timeout=remaining)
        connection.sock = current
        request_path = quote(parts.path or "/", safe="/%:@!$&'()*+,;=-._~")
        if parts.query:
            request_path += "?" + quote(parts.query, safe="%=&?/:@!$'()*+,;~-._")
        request_headers = {"User-Agent": "FuturePredictionBench/0.1", "Accept-Encoding": "identity", "Connection": "close"}
        request_headers.update(headers or {})
        connection.request(method, request_path, body=body, headers=request_headers)
        response = connection.getresponse()
        response_headers = {key.lower(): value for key, value in response.getheaders()}
        if response_headers.get("content-encoding", "identity").lower() != "identity":
            raise ProviderError("Compressed response unsupported")
        length = response_headers.get("content-length")
        if length is not None and (not length.isdigit() or int(length) > max_bytes):
            raise ProviderError("Response size limit exceeded")
        data = response.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ProviderError("Response size limit exceeded")
        if time.monotonic() - started >= timeout:
            raise ProviderError("Request timeout")
        return {"status": response.status, "headers": response_headers, "body": data, "url": url}
    except ProviderError:
        raise
    except (OSError, http.client.HTTPException, ValueError):
        # Do not echo URLs, request headers, secrets, or server error bodies.
        raise ProviderError("HTTP request failed") from None
    finally:
        timer.cancel()
        if connection is not None:
            connection.close()
        abort()


def request_json(url, *, payload=None, headers=None, timeout=30.0, public_only=True):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    request_headers = {"Accept": "application/json"}
    if data is not None:
        request_headers["Content-Type"] = "application/json"
    request_headers.update(headers or {})
    response = _request_bytes(url, method="GET" if data is None else "POST", body=data,
                              headers=request_headers, timeout=timeout, public_only=public_only)
    if response["status"] != 200:
        raise ProviderError(f"API HTTP status {response['status']}")
    try:
        result = strict_json_loads(response["body"])
    except (ValueError, TypeError, UnicodeError):
        raise ProviderError("Invalid API JSON response") from None
    if not isinstance(result, dict):
        raise ProviderError("API response must be an object")
    return result


class _PageText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = []
        self.in_title = False
        self.title = []
        self.text = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "template"}:
            self.hidden.append(tag)
        if tag == "title":
            self.in_title = True

    def handle_endtag(self, tag):
        if tag in self.hidden:
            # Tolerate malformed nesting without exposing hidden content.
            self.hidden = self.hidden[:self.hidden.index(tag)]
        if tag == "title":
            self.in_title = False

    def handle_data(self, data):
        if not self.hidden:
            self.text.append(data)
            if self.in_title:
                self.title.append(data)


def _plain_html(value):
    parser = _PageText()
    parser.feed(value)
    return " ".join(" ".join(parser.title).split()), " ".join(" ".join(parser.text).split())


class PublicPageReader:
    """Fetch text/HTML from public HTTP(S) on ports 80/443, without scripts."""

    def __init__(self, *, timeout=20.0, max_bytes=1_000_000, max_text_chars=16_000, max_redirects=4, transport=None):
        self.timeout = positive_seconds(timeout, "timeout")
        self.max_bytes = positive_int(max_bytes, "max_bytes", 10_000_000)
        self.max_text_chars = positive_int(max_text_chars, "max_text_chars", 1_000_000)
        if isinstance(max_redirects, bool) or not isinstance(max_redirects, int) or not 0 <= max_redirects <= 10:
            raise ValueError("max_redirects must be between 0 and 10")
        self.max_redirects = max_redirects
        self._transport = transport or _request_bytes

    def public_config(self):
        return {"adapter": "public_text_reader_v1", "timeout": self.timeout, "max_bytes": self.max_bytes,
                "max_text_chars": self.max_text_chars, "max_redirects": self.max_redirects}

    def open(self, url, *, timeout=None):
        deadline = time.monotonic() + min(self.timeout, positive_seconds(timeout, "timeout") if timeout is not None else self.timeout)
        for index in range(self.max_redirects + 1):
            # Validate every redirect even with an injected test transport.
            parts, hostname, port = _url_parts(url, True)
            try:
                literal = ipaddress.ip_address(hostname)
            except ValueError:
                literal = None
            if literal is not None and not _public_address(literal):
                raise ProviderError("Non-public destination blocked")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProviderError("Page request timeout")
            response = self._transport(url, timeout=remaining, max_bytes=self.max_bytes, public_only=True)
            if response["status"] in {301, 302, 303, 307, 308}:
                location = response["headers"].get("location")
                if index == self.max_redirects or not location:
                    raise ProviderError("Redirect limit or missing location")
                url = urljoin(url, location)
                continue
            if response["status"] != 200:
                raise ProviderError(f"Page HTTP status {response['status']}")
            mime = response["headers"].get("content-type", "").split(";", 1)[0].strip().lower()
            if mime not in {"text/html", "application/xhtml+xml", "text/plain"}:
                raise ProviderError("Unsupported page content type")
            body = response["body"]
            if len(body) > self.max_bytes:
                raise ProviderError("Response size limit exceeded")
            # UTF-8 replacement is deterministic; no browser encoding heuristics.
            text = body.decode("utf-8", errors="replace")
            title, text = _plain_html(text) if mime != "text/plain" else (hostname, text)
            return {"url": urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, "")),
                    "title": title[:1000] or hostname, "text": text[:self.max_text_chars]}
        raise ProviderError("Redirect limit exceeded")


class BraveResearchProvider:
    """Brave Web Search snippets plus a separate public webpage reader."""

    def __init__(self, api_key, *, count=5, timeout=20.0, reader=None, request_json=None):
        if not isinstance(api_key, str) or not api_key.strip() or any(c.isspace() for c in api_key):
            raise ValueError("FPB_BRAVE_API_KEY is required")
        self._api_key = api_key
        self.count = positive_int(count, "count", 20)
        self.timeout = positive_seconds(timeout, "timeout")
        self.reader = reader or PublicPageReader(timeout=timeout)
        self._request_json = request_json or globals()["request_json"]
        self.deadline = None

    @classmethod
    def from_env(cls):
        return cls(os.environ.get("FPB_BRAVE_API_KEY", ""))

    def public_config(self):
        return {"adapter": "brave_web_search_v1", "count": self.count, "timeout": self.timeout,
                "reader": self.reader.public_config()}

    def set_deadline(self, deadline):
        self.deadline = deadline

    def _timeout(self):
        remaining = self.timeout if self.deadline is None else min(self.timeout, self.deadline - time.monotonic())
        if remaining <= 0:
            raise ProviderError("Research wall time budget exhausted")
        return remaining

    def search(self, query):
        if not isinstance(query, str) or not query.strip() or len(query) > 600 or len(query.split()) > 75:
            raise ProviderError("Search query must contain at most 600 characters and 75 words")
        parameters = urlencode({"q": query, "count": self.count, "extra_snippets": "true"})
        response = self._request_json("https://api.search.brave.com/res/v1/web/search?" + parameters,
                                      headers={"X-Subscription-Token": self._api_key}, timeout=self._timeout())
        web = response.get("web", {})
        if not isinstance(web, dict):
            raise ProviderError("Invalid search response")
        raw_results = web.get("results", [])
        if not isinstance(raw_results, list):
            raise ProviderError("Invalid search response")
        results = []
        for item in raw_results[:self.count]:
            if not isinstance(item, dict) or not all(isinstance(item.get(k), str) for k in ("url", "title", "description")):
                continue
            extra = item.get("extra_snippets", [])
            if not isinstance(extra, list):
                extra = []
            snippet = item["description"] + "\n" + "\n".join(value for value in extra[:5] if isinstance(value, str))
            _, title = _plain_html(item["title"])
            _, text = _plain_html(snippet)
            # Relative ages are not verified publication timestamps.
            results.append({"url": item["url"], "title": title[:1000], "text": text[:8000]})
        return results

    def open(self, url):
        return self.reader.open(url, timeout=self._timeout())


class ChatCompletionModel:
    """A provider-neutral, nonstreaming /chat/completions adapter.

    The endpoint is trusted operator configuration and cannot be selected by the
    model. HTTPS is required except for a loopback development server.
    """

    def __init__(self, base_url, api_key, model_name, *, temperature=0.0, timeout=30.0, request_json=None):
        parts, hostname, port = _url_parts(base_url, False)
        if parts.query or parts.fragment:
            raise ValueError("Model base URL cannot contain query parameters or fragments")
        if parts.scheme != "https":
            try:
                loopback = ipaddress.ip_address(hostname).is_loopback
            except ValueError:
                loopback = hostname == "localhost"
            if not loopback:
                raise ValueError("Model endpoint requires HTTPS outside loopback")
        if not isinstance(api_key, str) or not api_key.strip() or any(c.isspace() for c in api_key):
            raise ValueError("FPB_MODEL_API_KEY is required; use a dummy value for an unauthenticated local server")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("FPB_MODEL_NAME is required")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError("temperature must be finite and in [0, 2]")
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.temperature = float(temperature)
        self.timeout = positive_seconds(timeout, "timeout")
        self._api_key = api_key
        self._request_json = request_json or globals()["request_json"]

    @classmethod
    def from_env(cls):
        names = ("FPB_MODEL_BASE_URL", "FPB_MODEL_API_KEY", "FPB_MODEL_NAME")
        missing = [name for name in names if not os.environ.get(name, "").strip()]
        if missing:
            raise ValueError("Missing model configuration: " + ", ".join(missing))
        return cls(*(os.environ[name] for name in names))

    def public_config(self):
        return {"adapter": "chat_completions_v1", "base_url": self.base_url, "model": self.model_name,
                "temperature": self.temperature, "timeout": self.timeout}

    def complete(self, messages, tools, *, max_tokens=1024, timeout=None):
        positive_int(max_tokens, "max_tokens")
        effective_timeout = min(self.timeout, positive_seconds(timeout, "timeout") if timeout is not None else self.timeout)
        payload = {"model": self.model_name, "messages": messages, "temperature": self.temperature,
                   "max_tokens": max_tokens, "stream": False}
        if tools:
            payload.update(tools=tools, tool_choice="auto", parallel_tool_calls=False)
        return self._request_json(self.base_url + "/chat/completions", payload=payload,
                                  headers={"Authorization": "Bearer " + self._api_key},
                                  timeout=effective_timeout, public_only=False)
