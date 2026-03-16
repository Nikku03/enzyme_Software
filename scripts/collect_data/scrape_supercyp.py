from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://insilico-cyp.charite.de/SuperCYPsPred"
MAJOR_CYPS = ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"]
ALL_CYPS = MAJOR_CYPS + ["CYP2A6", "CYP2B6", "CYP2C8", "CYP2E1", "CYP3A5", "CYP2C18", "CYP46A"]
SEARCH_TERMS = list("abcdefghijklmnopqrstuvwxyz") + [
    "ace", "ami", "azi", "cef", "clo", "dia", "ery", "flu", "ibu", "ket", "lor", "met", "nif", "ome",
    "par", "qui", "ran", "sim", "tam", "war",
]
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/json,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
REQUEST_RETRIES = 4
BACKOFF_BASE_S = 1.5

CYP_COLUMN_MAP = {
    "1A1": "CYP1A1",
    "1A2": "CYP1A2",
    "2A6": "CYP2A6",
    "2B6": "CYP2B6",
    "2C8": "CYP2C8",
    "2C9": "CYP2C9",
    "2C18": "CYP2C18",
    "2C19": "CYP2C19",
    "2D6": "CYP2D6",
    "2E1": "CYP2E1",
    "3A4": "CYP3A4",
    "3A5": "CYP3A5",
    "46A": "CYP46A",
}


def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip()).strip()


def _normalize_supercyp_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _normalize_name(name).lower())


def _build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    retry = Retry(
        total=REQUEST_RETRIES,
        connect=REQUEST_RETRIES,
        read=REQUEST_RETRIES,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _request(session: requests.Session, method: str, url: str, timeout: int = 30, **kwargs) -> requests.Response:
    last_exc = None
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            response = session.request(method, url, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_exc = exc
            if attempt == REQUEST_RETRIES:
                break
            sleep_s = BACKOFF_BASE_S * attempt
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


def parse_autocomplete_options(html: str) -> List[str]:
    options = re.findall(r'<option value="([^"]+)"', html, flags=re.IGNORECASE)
    deduped: List[str] = []
    seen = set()
    for option in options:
        value = _normalize_name(option)
        if value and value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def search_drug_autocomplete(query: str, session: Optional[requests.Session] = None, timeout: int = 20) -> List[str]:
    sess = session or _build_session()
    where = f" synonyms like '%{query}%' "
    search = f"select count(distinct cas) from drugs1 WHERE {where}"
    response = _request(
        sess,
        "GET",
        f"{BASE_URL}/livesearch.php",
        params={"search": search, "where": where},
        timeout=timeout,
    )
    return parse_autocomplete_options(response.text)


def _strip_tags(fragment: str) -> str:
    text = re.sub(r"<script.*?</script>", " ", fragment, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_tables(html: str) -> List[List[List[str]]]:
    tables: List[List[List[str]]] = []
    for table_html in re.findall(r"<table.*?</table>", html, flags=re.IGNORECASE | re.DOTALL):
        rows: List[List[str]] = []
        for row_html in re.findall(r"<tr.*?</tr>", table_html, flags=re.IGNORECASE | re.DOTALL):
            cells = re.findall(r"<(?:td|th)[^>]*>(.*?)</(?:td|th)>", row_html, flags=re.IGNORECASE | re.DOTALL)
            parsed = [_strip_tags(cell) for cell in cells]
            if parsed:
                rows.append(parsed)
        if rows:
            tables.append(rows)
    return tables


def parse_interaction_table(html: str, drug_name: str) -> Optional[Dict[str, object]]:
    target_norm = _normalize_supercyp_name(drug_name)
    for rows in _extract_tables(html):
        headers = [re.sub(r"\s+Cytochrome.*$", "", cell).strip() for cell in rows[0]]
        if not headers or "Name" not in headers:
            continue
        for row_values in rows[1:]:
            if not row_values:
                continue
            row_name = row_values[0]
            if not row_name or row_name.lower().startswith("alternative drugs"):
                continue
            row_norm = _normalize_supercyp_name(row_name)
            if target_norm not in row_norm and row_norm not in target_norm:
                continue
            interactions = {
                "name": _normalize_name(drug_name),
                "cyp_substrate": [],
                "cyp_inhibitor": [],
                "cyp_inducer": [],
            }
            for idx, header in enumerate(headers[1:], start=1):
                if idx >= len(row_values):
                    break
                cyp = CYP_COLUMN_MAP.get(header)
                if cyp is None:
                    continue
                cell = row_values[idx].lower()
                if re.search(r"\bs\b", cell):
                    interactions["cyp_substrate"].append(cyp)
                if "inh" in cell and "p-inh" not in cell:
                    interactions["cyp_inhibitor"].append(cyp)
                if re.search(r"\bind\b", cell):
                    interactions["cyp_inducer"].append(cyp)
            if interactions["cyp_substrate"] or interactions["cyp_inhibitor"] or interactions["cyp_inducer"]:
                for key in ("cyp_substrate", "cyp_inhibitor", "cyp_inducer"):
                    interactions[key] = sorted(set(interactions[key]))
                return interactions
    return None


def get_drug_interactions(drug_name: str, session: Optional[requests.Session] = None, timeout: int = 30) -> Optional[Dict[str, object]]:
    sess = session or _build_session()
    response = _request(
        sess,
        "POST",
        f"{BASE_URL}/index.php?site=get_drug_interaction",
        data={"textfeld": f"{drug_name}\n"},
        timeout=timeout,
    )
    return parse_interaction_table(response.text, drug_name)


def get_smiles_from_pubchem(drug_name: str, session: Optional[requests.Session] = None, timeout: int = 20) -> Optional[str]:
    sess = session or _build_session()
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"
    for prop in ("CanonicalSMILES", "ConnectivitySMILES"):
        try:
            response = _request(sess, "GET", f"{base}/{requests.utils.quote(drug_name)}/property/{prop}/JSON", timeout=timeout)
        except requests.RequestException:
            continue
        try:
            props = response.json().get("PropertyTable", {}).get("Properties", [])
        except Exception:
            continue
        if props:
            value = props[0].get(prop)
            if value:
                return str(value)
    return None


def choose_primary_cyp(cyps: Iterable[str]) -> Optional[str]:
    cyps = [str(c) for c in cyps if c]
    for cyp in MAJOR_CYPS:
        if cyp in cyps:
            return cyp
    return cyps[0] if cyps else None


def collect_supercyp_drugs(
    output_path: str = "data/supercyp_drugs.json",
    search_terms: Optional[List[str]] = None,
    sleep_s: float = 0.3,
) -> List[Dict[str, object]]:
    print("=" * 60)
    print("Collecting Drugs from SuperCYP")
    print("=" * 60)
    terms = search_terms or SEARCH_TERMS
    session = _build_session()
    collected: List[Dict[str, object]] = []
    seen_names = set()
    out = Path(output_path)

    def save_progress() -> None:
        deduped_tmp: List[Dict[str, object]] = []
        by_smiles_tmp: Dict[str, Dict[str, object]] = {}
        for drug in collected:
            smiles = str(drug["smiles"])
            if smiles not in by_smiles_tmp:
                by_smiles_tmp[smiles] = drug
                deduped_tmp.append(drug)
                continue
            existing = by_smiles_tmp[smiles]
            merged = sorted(set(existing.get("all_cyp_substrates", [])) | set(drug.get("all_cyp_substrates", [])))
            existing["all_cyp_substrates"] = merged
            existing["primary_cyp"] = choose_primary_cyp(merged)
        by_cyp_tmp = Counter(str(d["primary_cyp"]) for d in deduped_tmp)
        payload_tmp = {
            "metadata": {
                "source": "SuperCYP",
                "url": f"{BASE_URL}/",
                "total_drugs": len(deduped_tmp),
                "cyp_distribution": dict(by_cyp_tmp),
            },
            "drugs": deduped_tmp,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload_tmp, indent=2))

    for term in terms:
        print(f"Searching: {term}...")
        try:
            candidates = search_drug_autocomplete(term, session=session)
        except Exception as exc:
            print(f"  search failed: {exc}")
            continue
        for name in candidates:
            normalized = _normalize_name(name)
            if not normalized or normalized in seen_names:
                continue
            seen_names.add(normalized)
            try:
                interactions = get_drug_interactions(normalized, session=session)
            except Exception as exc:
                print(f"  interaction lookup failed for {normalized}: {exc}")
                continue
            if not interactions or not interactions["cyp_substrate"]:
                continue
            try:
                smiles = get_smiles_from_pubchem(normalized, session=session)
            except Exception as exc:
                print(f"  pubchem lookup failed for {normalized}: {exc}")
                continue
            if not smiles:
                continue
            primary_cyp = choose_primary_cyp(interactions["cyp_substrate"])
            if not primary_cyp:
                continue
            entry = {
                "name": normalized,
                "smiles": smiles,
                "primary_cyp": primary_cyp,
                "all_cyp_substrates": list(interactions["cyp_substrate"]),
                "cyp_inhibitor": list(interactions["cyp_inhibitor"]),
                "cyp_inducer": list(interactions["cyp_inducer"]),
                "source": "SuperCYP",
                "confidence": "validated",
                "cyp_label_source": "SuperCYP",
            }
            collected.append(entry)
            print(f"  Added: {normalized} ({primary_cyp})")
            if len(collected) % 25 == 0:
                save_progress()
            time.sleep(sleep_s)
        time.sleep(min(0.5, sleep_s))
    save_progress()
    deduped: List[Dict[str, object]] = []
    by_smiles: Dict[str, Dict[str, object]] = {}
    for drug in collected:
        smiles = str(drug["smiles"])
        if smiles not in by_smiles:
            by_smiles[smiles] = drug
            deduped.append(drug)
            continue
        existing = by_smiles[smiles]
        merged = sorted(set(existing.get("all_cyp_substrates", [])) | set(drug.get("all_cyp_substrates", [])))
        existing["all_cyp_substrates"] = merged
        existing["primary_cyp"] = choose_primary_cyp(merged)
    by_cyp = Counter(str(d["primary_cyp"]) for d in deduped)
    payload = {
        "metadata": {
            "source": "SuperCYP",
            "url": f"{BASE_URL}/",
            "total_drugs": len(deduped),
            "cyp_distribution": dict(by_cyp),
        },
        "drugs": deduped,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nTotal drugs collected: {len(deduped)}")
    print("\nCYP Distribution:")
    for cyp, count in sorted(by_cyp.items()):
        print(f"  {cyp}: {count}")
    print(f"\nSaved to {out}")
    return deduped


if __name__ == "__main__":
    collect_supercyp_drugs()
