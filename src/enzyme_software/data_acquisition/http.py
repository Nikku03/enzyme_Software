from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen


KNOWN_API_HOSTS = {"pubchem.ncbi.nlm.nih.gov", "www.ebi.ac.uk"}


@dataclass
class RateLimiter:
    tokens_per_second: float
    capacity: float
    tokens: float
    last_refill: float

    def acquire(self, amount: float = 1.0) -> None:
        while True:
            self._refill()
            if self.tokens >= amount:
                self.tokens -= amount
                return
            time.sleep(max(0.01, 1.0 / max(self.tokens_per_second, 1e-6)))

    def _refill(self) -> None:
        now = time.time()
        elapsed = max(0.0, now - self.last_refill)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.tokens_per_second)
        self.last_refill = now


class HttpClient:
    def __init__(
        self,
        cache_dir: str = ".cache",
        tokens_per_second: float = 1.0,
        capacity: float = 3.0,
        timeout_s: float = 15.0,
        retries: int = 2,
        backoff_s: float = 1.0,
        user_agent: str = "enzyme_software/1.0",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limiter = RateLimiter(
            tokens_per_second=tokens_per_second,
            capacity=capacity,
            tokens=capacity,
            last_refill=time.time(),
        )
        self.timeout_s = timeout_s
        self.retries = retries
        self.backoff_s = backoff_s
        self.user_agent = user_agent
        self._robots_cache: dict[str, Optional[str]] = {}
        self._whitelist = _load_whitelist()

    def get(self, url: str, use_cache: bool = True) -> bytes:
        cache_path = self._cache_path(url)
        if use_cache and cache_path.exists():
            return cache_path.read_bytes()

        parsed = urlparse(url)
        host = parsed.netloc
        if not host:
            raise ValueError("Invalid URL: missing host.")
        if host not in KNOWN_API_HOSTS and host not in self._whitelist:
            raise PermissionError(f"Domain not whitelisted: {host}")

        if not self.allowed_by_robots(url, self.user_agent):
            raise PermissionError("Blocked by robots.txt")

        last_error: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                self.rate_limiter.acquire()
                request = Request(url, headers={"User-Agent": self.user_agent})
                with urlopen(request, timeout=self.timeout_s) as response:
                    payload = response.read()
                if use_cache:
                    cache_path.write_bytes(payload)
                return payload
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(self.backoff_s * (2**attempt))
        raise RuntimeError(f"HTTP fetch failed: {last_error}")

    def allowed_by_robots(self, url: str, user_agent: str = "*") -> bool:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False
        if parsed.netloc in KNOWN_API_HOSTS:
            return True
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rules = self._robots_cache.get(robots_url)
        if rules is None:
            try:
                rules = self._fetch_robots(robots_url)
            except Exception:
                rules = ""
            self._robots_cache[robots_url] = rules
        return _robots_allows(rules or "", user_agent, parsed.path or "/")

    def _fetch_robots(self, robots_url: str) -> str:
        request = Request(robots_url, headers={"User-Agent": self.user_agent})
        with urlopen(request, timeout=self.timeout_s) as response:
            return response.read().decode("utf-8", errors="replace")

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.cache"


def _robots_allows(rules_text: str, user_agent: str, path: str) -> bool:
    rules = _parse_robots(rules_text)
    agent = user_agent.lower()
    applicable = rules.get(agent)
    if applicable is None:
        applicable = rules.get("*", [])
    if not applicable:
        return True
    for rule in applicable:
        if path.startswith(rule):
            return False
    return True


def _parse_robots(text: str) -> dict[str, list[str]]:
    rules: dict[str, list[str]] = {}
    current_agent: Optional[str] = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        key_lower = key.lower()
        if key_lower == "user-agent":
            current_agent = value.lower()
            rules.setdefault(current_agent, [])
        elif key_lower == "disallow" and current_agent is not None:
            rules[current_agent].append(value or "/")
    return rules


def _load_whitelist() -> set[str]:
    repo_root = Path(__file__).resolve().parents[3]
    whitelist_path = repo_root / "data_acquisition" / "whitelist.txt"
    if not whitelist_path.is_file():
        return set()
    domains: set[str] = set()
    for line in whitelist_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        domains.add(entry)
    return domains
