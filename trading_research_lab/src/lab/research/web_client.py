from __future__ import annotations

import logging
from typing import Iterable, List

import requests

LOGGER = logging.getLogger(__name__)


def fetch_url(url: str, timeout: int = 30) -> bytes:
    LOGGER.info("Fetching URL: %s", url)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def build_search_urls(keywords: Iterable[str]) -> List[str]:
    query = "+".join(k.strip().replace(" ", "+") for k in keywords if k.strip())
    if not query:
        return []
    return [
        f"https://duckduckgo.com/?q={query}",
        f"https://www.google.com/search?q={query}",
    ]
