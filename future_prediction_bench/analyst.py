"""Versioned, bounded analyst tools; no shell, filesystem or private-store access."""

from __future__ import annotations

import ast
import copy
import json
import math
import operator
import re
import time
from urllib.parse import parse_qs, urlencode, urlsplit

from .providers import ProviderError, positive_seconds, request_json, strict_json_loads
from .schema import option_ids, validate_probabilities

TOOLKIT_VERSION = "analyst_tools_v1"
RESEARCH_TOOLS = frozenset({"search", "open", "calculator", "notebook", "draft", "market_search", "market_snapshot"})
MARKET_TOOLS = frozenset({"market_search", "market_snapshot"})


def _string(maximum, minimum=1):
    return {"type": "string", "minLength": minimum, "maxLength": maximum}


def tool_definitions(question, research_mode="self_research", market_mode="no_consensus"):
    """The same versioned schemas drive model discovery and runtime validation."""
    distribution = {"type": "object", "properties": {key: {"type": "number", "minimum": 0, "maximum": 1}
                    for key in option_ids(question)}, "required": option_ids(question), "additionalProperties": False}
    sources = {"type": "array", "items": _string(64, 64), "minItems": 0, "maxItems": 20, "uniqueItems": True}
    definitions = [("submit", "Submit the final distribution for every option ID.", {"probabilities": distribution})]
    if research_mode == "self_research":
        definitions = [
            ("search", "Search the public web for evidence; at least one successful web search is required.", {"query": _string(600)}),
            ("open", "Read text from one public HTTP(S) webpage.", {"url": _string(8192)}),
            ("calculator", "Evaluate bounded arithmetic with + - * / ** and parentheses; no code or functions.", {"expression": _string(512)}),
            ("notebook", "Append an evidence claim using hashes from successful source observations in this episode. Does not verify that a source supports a claim.",
             {"claim": _string(2000), "source_hashes": {**sources, "minItems": 1}}),
            ("draft", "Record a tentative full probability distribution and its revision rationale. This is not submission and receives no outcome reward.",
             {"probabilities": distribution, "rationale": _string(4000), "source_hashes": sources}),
        ] + definitions
        if market_mode == "market_aware":
            definitions[-1:-1] = [
                ("market_search", "Discover public Polymarket markets. Candidates are not verified matches to the benchmark question.", {"query": _string(300)}),
                ("market_snapshot", "Read public Polymarket metadata and reported outcome prices for a numeric market ID. Prices are not guaranteed current or a matched benchmark baseline.", {"market_id": {**_string(32), "pattern": "^[0-9]+$"}}),
            ]
    return [{"type": "function", "function": {"name": name, "description": description,
             "parameters": {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}}}
            for name, description, fields in definitions]


def _validate(value, schema):
    kind = schema["type"]
    if kind == "object":
        if not isinstance(value, dict) or set(value) != set(schema["required"]):
            raise ValueError("invalid_tool_arguments")
        for key, item in value.items():
            _validate(item, schema["properties"][key])
    elif kind == "string":
        if (not isinstance(value, str) or not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", 10000)
                or not value.strip() or (schema.get("pattern") and not re.fullmatch(schema["pattern"], value))):
            raise ValueError("invalid_tool_arguments")
    elif kind == "array":
        if not isinstance(value, list) or not schema["minItems"] <= len(value) <= schema["maxItems"]:
            raise ValueError("invalid_tool_arguments")
        for item in value:
            _validate(item, schema["items"])
        if len(set(value)) != len(value):
            raise ValueError("invalid_tool_arguments")
    elif kind == "number":
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not schema["minimum"] <= value <= schema["maximum"]:
            raise ValueError("invalid_tool_arguments")


def validate_arguments(name, arguments, question, research_mode="self_research", market_mode="no_consensus"):
    definitions = {entry["function"]["name"]: entry["function"]["parameters"]
                   for entry in tool_definitions(question, research_mode, market_mode)}
    if name not in definitions:
        raise ValueError("tool_not_available")
    # Final distribution validation belongs to Store.submit so malformed forecasts
    # remain immutable, auditable protocol failures rather than implicit retries.
    if name == "submit":
        if not isinstance(arguments, dict) or set(arguments) != {"probabilities"}:
            raise ValueError("invalid_tool_arguments")
    else:
        _validate(arguments, definitions[name])


def calculate(expression):
    """Interpret only finite numeric arithmetic; never call eval or compile."""
    if not isinstance(expression, str) or not 1 <= len(expression) <= 512:
        raise ValueError("invalid_expression")
    try:
        tree = ast.parse(expression, mode="eval")
        if len(list(ast.walk(tree))) > 100:
            raise ValueError("expression_too_complex")
        operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                      ast.Div: operator.truediv, ast.Pow: operator.pow}

        def number(value):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 1e100:
                raise ValueError("arithmetic_range_exceeded")
            return float(value)

        def visit(node, depth=0):
            if depth > 20:
                raise ValueError("expression_too_complex")
            if isinstance(node, ast.Constant):
                return number(node.value)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                return number(visit(node.operand, depth + 1) * (-1 if isinstance(node.op, ast.USub) else 1))
            if isinstance(node, ast.BinOp) and type(node.op) in operations:
                left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
                if isinstance(node.op, ast.Pow) and abs(right) > 100:
                    raise ValueError("exponent_out_of_range")
                return number(operations[type(node.op)](left, right))
            raise ValueError("unsupported_expression")

        return visit(tree.body)
    except (SyntaxError, TypeError, OverflowError, ZeroDivisionError, RecursionError):
        raise ValueError("invalid_expression") from None


class AnalystNotebook:
    """Episode-local agent-authored notes and drafts, with source provenance only."""

    def __init__(self, question):
        self.question = copy.deepcopy(question)
        self.notes = []
        self.drafts = []

    @staticmethod
    def _sources(hashes, sources):
        if any(value not in sources for value in hashes):
            raise ValueError("unknown_source_hash")
        return [{"sha256": value, "url": sources[value]["url"]} for value in hashes]

    def note(self, claim, source_hashes, sources):
        if len(self.notes) >= 100:
            raise ValueError("notebook_full")
        note = {"note_id": len(self.notes) + 1, "claim": claim,
                "sources": self._sources(source_hashes, sources), "authorship": "agent", "support_verified": False}
        self.notes.append(copy.deepcopy(note))
        return note

    def draft(self, probabilities, rationale, source_hashes, sources):
        if len(self.drafts) >= 100:
            raise ValueError("draft_limit_reached")
        probabilities = validate_probabilities(probabilities, option_ids(self.question))
        draft = {"revision": len(self.drafts) + 1, "probabilities": probabilities, "rationale": rationale,
                 "sources": self._sources(source_hashes, sources), "authorship": "agent", "submitted": False,
                 "previous_probabilities": copy.deepcopy(self.drafts[-1]["probabilities"]) if self.drafts else None}
        self.drafts.append(copy.deepcopy(draft))
        return draft


class PolymarketPublicProvider:
    """Public GET-only Gamma API; entirely separate from private baseline storage.

    The destination is an operator-owned, exact HTTPS allowlist. System DNS may
    route this fixed service through a managed network. TLS hostname validation,
    bounded responses and redirect rejection still apply. This transport must
    never be used for model-selected URLs in the generic webpage reader.

    Official references: docs.polymarket.com/api-reference/search/search-markets-events-and-profiles
    and docs.polymarket.com/api-reference/markets/get-market-by-id.
    """

    def __init__(self, *, timeout=20.0, request=None):
        self.timeout = positive_seconds(timeout, "timeout")
        self._request = request or request_json
        self.deadline = None

    def public_config(self):
        return {"adapter": "polymarket_public_gamma_v1", "timeout": self.timeout, "max_markets": 5,
                "transport": "fixed_gamma_https_get_v1", "destination_host": "gamma-api.polymarket.com",
                "dns_policy": "system_dns_for_fixed_host", "redirects": "reject"}

    def set_deadline(self, deadline):
        self.deadline = deadline

    def _timeout(self, timeout):
        result = min(self.timeout, timeout if timeout is not None else self.timeout)
        if self.deadline is not None:
            result = min(result, self.deadline - time.monotonic())
        if result <= 0:
            raise ProviderError("Research wall time budget exhausted")
        return result

    def _get(self, url, *, timeout=None):
        """Constrain the whole route before using trusted fixed-host transport."""
        try:
            if not isinstance(url, str) or len(url) > 8192 or "\\" in url or any(c.isspace() or ord(c) < 32 for c in url):
                raise ValueError()
            parsed = urlsplit(url)
            if parsed.scheme != "https" or parsed.netloc != "gamma-api.polymarket.com" or parsed.fragment:
                raise ValueError()
            if parsed.path == "/public-search":
                query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
                required = {"q", "limit_per_type", "search_profiles", "search_tags"}
                if set(query) != required or any(len(values) != 1 for values in query.values()):
                    raise ValueError()
                if (not 1 <= len(query["q"][0]) <= 300 or not query["q"][0].strip()
                        or query["limit_per_type"] != ["5"] or query["search_profiles"] != ["false"]
                        or query["search_tags"] != ["false"]):
                    raise ValueError()
            elif not re.fullmatch(r"/markets/[0-9]{1,32}", parsed.path) or parsed.query:
                raise ValueError()
        except (ValueError, TypeError):
            raise ProviderError("Market URL is outside the fixed HTTPS route allowlist") from None
        # request_json only issues GET without payload, verifies TLS for the
        # original hostname and rejects non-200 responses without redirects.
        return self._request(url, timeout=self._timeout(timeout), public_only=False)

    @staticmethod
    def _market(raw):
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), str) or not re.fullmatch(r"[0-9]{1,32}", raw["id"]):
            raise ProviderError("Invalid market response")
        if not isinstance(raw.get("question"), str):
            raise ProviderError("Invalid market response")
        result = {"market_id": raw["id"], "question": raw["question"][:2000],
                  "description": raw.get("description", "")[:8000] if isinstance(raw.get("description", ""), str) else "",
                  "match_verified": False, "price_semantics": "provider_reported_outcome_prices_not_normalized"}
        for key in ("endDate", "updatedAt", "resolutionSource", "active", "closed", "archived", "bestBid", "bestAsk", "lastTradePrice"):
            value = raw.get(key)
            if isinstance(value, (str, bool)) or (isinstance(value, (int, float)) and math.isfinite(value)):
                result[key] = value[:2000] if isinstance(value, str) else value
        for key in ("outcomes", "outcomePrices"):
            value = raw.get(key)
            try:
                value = strict_json_loads(value) if isinstance(value, str) else value
            except (ValueError, TypeError):
                raise ProviderError("Invalid market outcomes") from None
            if not isinstance(value, list) or not 2 <= len(value) <= 20:
                raise ProviderError("Invalid market outcomes")
            if key == "outcomes":
                if any(not isinstance(item, str) or not item or len(item) > 500 for item in value):
                    raise ProviderError("Invalid market outcomes")
                result[key] = value
            else:
                try:
                    if any(isinstance(item, bool) for item in value):
                        raise ValueError()
                    values = [float(item) for item in value]
                    if any(not math.isfinite(item) or not 0 <= item <= 1 for item in values):
                        raise ValueError()
                except (ValueError, TypeError, OverflowError):
                    raise ProviderError("Invalid market prices") from None
                result[key] = values
        if len(result["outcomes"]) != len(result["outcomePrices"]):
            raise ProviderError("Market outcome/price mismatch")
        return {"url": "https://gamma-api.polymarket.com/markets/" + raw["id"],
                "title": result["question"], "text": json.dumps(result, ensure_ascii=False, allow_nan=False),
                "source_updated_at": raw.get("updatedAt"), "market": result}

    def search(self, query, *, timeout=None):
        if not isinstance(query, str) or not 1 <= len(query) <= 300 or not query.strip():
            raise ValueError("invalid_market_query")
        url = "https://gamma-api.polymarket.com/public-search?" + urlencode({"q": query, "limit_per_type": 5, "search_profiles": "false", "search_tags": "false"})
        result = self._get(url, timeout=timeout)
        events = result.get("events") if isinstance(result, dict) else None
        if not isinstance(events, list):
            raise ProviderError("Invalid market search response")
        markets, seen = [], set()
        for event in events[:5]:
            if not isinstance(event, dict) or not isinstance(event.get("markets"), list):
                continue
            for raw in event["markets"][:100]:
                try:
                    market = self._market(raw)
                except ProviderError:
                    continue
                identifier = market["market"]["market_id"]
                if identifier not in seen:
                    markets.append(market)
                    seen.add(identifier)
                if len(markets) == 5:
                    return markets
        return markets

    def snapshot(self, market_id, *, timeout=None):
        if not isinstance(market_id, str) or not re.fullmatch(r"[0-9]{1,32}", market_id):
            raise ValueError("invalid_market_id")
        raw = self._get("https://gamma-api.polymarket.com/markets/" + market_id, timeout=timeout)
        result = self._market(raw)
        if result["market"]["market_id"] != market_id:
            raise ProviderError("Market identity mismatch")
        return result
