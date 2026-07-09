"""Web/arxiv lookup for the Research agent, behind one pluggable, offline-safe API.

A single :func:`web_search` fronts several backends (Tavily / SerpAPI / DuckDuckGo).
Every function degrades gracefully: with no network or no configured backend it
returns an empty result set plus a ``note`` rather than raising, so the
formulas-provided branch and the test suite never depend on connectivity.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Optional

_USER_AGENT = "pinnsystem-research/0.1"
_TIMEOUT = 10.0


def _http_get(url: str, *, timeout: float = _TIMEOUT) -> Optional[bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed schemes
            return resp.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def web_search(
    query: str,
    *,
    backend: str = "duckduckgo",
    api_key: Optional[str] = None,
    max_results: int = 5,
) -> dict[str, Any]:
    """Search the web, returning ``{results: [{title, url, snippet}], note}``."""

    if backend == "none":
        return {"results": [], "note": "search disabled (backend=none)"}

    if backend == "tavily" and api_key:
        return _tavily(query, api_key, max_results)
    if backend == "serpapi" and api_key:
        return _serpapi(query, api_key, max_results)

    # Default: DuckDuckGo instant-answer API — keyless, best-effort.
    return _duckduckgo(query, max_results)


def _duckduckgo(query: str, max_results: int) -> dict[str, Any]:
    params = urllib.parse.urlencode({"q": query, "format": "json", "no_html": "1"})
    raw = _http_get(f"https://api.duckduckgo.com/?{params}")
    if raw is None:
        return {"results": [], "note": "offline or search unavailable"}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"results": [], "note": "malformed search response"}

    results: list[dict[str, str]] = []
    for topic in data.get("RelatedTopics", []):
        if "Text" in topic and "FirstURL" in topic:
            results.append(
                {
                    "title": topic["Text"][:120],
                    "url": topic["FirstURL"],
                    "snippet": topic["Text"],
                }
            )
        if len(results) >= max_results:
            break
    return {"results": results, "note": None if results else "no results"}


def _tavily(query: str, api_key: str, max_results: int) -> dict[str, Any]:
    payload = json.dumps(
        {"api_key": api_key, "query": query, "max_results": max_results}
    ).encode()
    req = urllib.request.Request(
        "https://api.tavily.com/search",
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": _USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:  # noqa: S310
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return {"results": [], "note": "tavily unavailable"}
    results = [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
        for r in data.get("results", [])
    ]
    return {"results": results, "note": None}


def _serpapi(query: str, api_key: str, max_results: int) -> dict[str, Any]:
    params = urllib.parse.urlencode({"q": query, "api_key": api_key, "num": max_results})
    raw = _http_get(f"https://serpapi.com/search.json?{params}")
    if raw is None:
        return {"results": [], "note": "serpapi unavailable"}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"results": [], "note": "malformed serpapi response"}
    results = [
        {"title": r.get("title", ""), "url": r.get("link", ""), "snippet": r.get("snippet", "")}
        for r in data.get("organic_results", [])[:max_results]
    ]
    return {"results": results, "note": None}


def fetch_url(url: str, *, max_chars: int = 20000) -> dict[str, Any]:
    """Fetch a URL's text content (truncated), or an error note."""

    raw = _http_get(url)
    if raw is None:
        return {"ok": False, "text": "", "note": "fetch failed (offline or bad URL)"}
    text = raw.decode("utf-8", errors="replace")
    return {"ok": True, "text": text[:max_chars], "note": None}


def arxiv_search(query: str, *, max_results: int = 5) -> dict[str, Any]:
    """Query the arXiv API for relevant papers (best-effort, offline-safe)."""

    params = urllib.parse.urlencode(
        {"search_query": f"all:{query}", "start": 0, "max_results": max_results}
    )
    raw = _http_get(f"http://export.arxiv.org/api/query?{params}")
    if raw is None:
        return {"results": [], "note": "arxiv unavailable"}
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return {"results": [], "note": "malformed arxiv response"}

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    results = []
    for entry in root.findall("atom:entry", ns):
        title = entry.findtext("atom:title", default="", namespaces=ns).strip()
        summary = entry.findtext("atom:summary", default="", namespaces=ns).strip()
        link = entry.findtext("atom:id", default="", namespaces=ns).strip()
        results.append({"title": title, "url": link, "snippet": summary[:400]})
    return {"results": results, "note": None if results else "no results"}
