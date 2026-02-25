import os
import re
import json
import time
import hashlib
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import requests
import feedparser
from dateutil import tz
from tqdm import tqdm

import google.generativeai as genai


# -----------------------------
# Config
# -----------------------------

LOCAL_TZ = tz.gettz("America/Los_Angeles")

COMPANIES = [
    {"ticker": "NVDA", "name": "NVIDIA", "query": "NVIDIA OR NVDA"},
    {"ticker": "MSFT", "name": "Microsoft", "query": "Microsoft OR MSFT"},
]

# Google News RSS (no API key). You can swap to other RSS sources easily.
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"

OUT_DIR = "reports"
CACHE_DIR = "cache"
MAX_ITEMS_PER_COMPANY = 30  # pull this many per company per run

# Light “reliability” heuristic: prefer mainstream domains if present.
TRUSTED_DOMAIN_HINTS = [
    "reuters.com", "bloomberg.com", "wsj.com", "ft.com", "nytimes.com",
    "theverge.com", "techcrunch.com", "arstechnica.com",
    "sec.gov", "microsoft.com", "nvidia.com", "investor", "ir.",
]

# Gemini config
# export GEMINI_API_KEY="..."
# export GEMINI_MODEL="gemini-1.5-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Configure Gemini once (recommended)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# -----------------------------
# Data model
# -----------------------------

@dataclass
class Article:
    company: str          # ticker
    title: str
    link: str
    published_utc: Optional[dt.datetime]
    source: str
    snippet: str = ""


@dataclass
class EventCluster:
    company: str
    event_id: str
    headline: str
    articles: List[Article]
    canonical_summary: str
    event_type: str
    importance: int       # 1-5
    confidence: int       # 1-5


# -----------------------------
# Utilities
# -----------------------------

def ensure_dirs() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 %\-\.\:/_]", "", s)
    return s


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def parse_published(entry) -> Optional[dt.datetime]:
    if getattr(entry, "published_parsed", None):
        t = time.mktime(entry.published_parsed)
        return dt.datetime.fromtimestamp(t, tz=dt.timezone.utc)
    if getattr(entry, "updated_parsed", None):
        t = time.mktime(entry.updated_parsed)
        return dt.datetime.fromtimestamp(t, tz=dt.timezone.utc)
    return None


def domain_from_url(url: str) -> str:
    m = re.match(r"^https?://([^/]+)/", url)
    return m.group(1).lower() if m else ""


def trust_score(url: str) -> int:
    d = domain_from_url(url)
    for hint in TRUSTED_DOMAIN_HINTS:
        if hint in d:
            return 2
    return 1


def to_local_time_str(dt_utc: Optional[dt.datetime]) -> str:
    if not dt_utc:
        return "Unknown time"
    local = dt_utc.astimezone(LOCAL_TZ)
    return local.strftime("%Y-%m-%d %H:%M %Z")


# -----------------------------
# Retrieval
# -----------------------------

def fetch_rss_articles(company: Dict[str, str]) -> List[Article]:
    q = requests.utils.quote(company["query"])
    url = GOOGLE_NEWS_RSS.format(q=q)

    rss_cache_path = os.path.join(CACHE_DIR, f"rss_{company['ticker']}.xml")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    with open(rss_cache_path, "wb") as f:
        f.write(resp.content)

    feed = feedparser.parse(resp.content)
    items: List[Article] = []

    for e in feed.entries[:MAX_ITEMS_PER_COMPANY]:
        title = getattr(e, "title", "").strip()
        link = getattr(e, "link", "").strip()

        if not title or not link:
            continue

        if getattr(e, "source", None) and getattr(e.source, "title", None):
            source = str(e.source.title).strip()
        else:
            source = domain_from_url(link) or "Unknown"

        published = parse_published(e)

        snippet = getattr(e, "summary", "") or ""
        snippet = re.sub(r"<[^>]+>", "", snippet).strip()

        items.append(Article(
            company=company["ticker"],
            title=title,
            link=link,
            published_utc=published,
            source=source,
            snippet=snippet[:280],
        ))

    return items


# -----------------------------
# Dedup / clustering (lightweight)
# -----------------------------

def simple_dedup(articles: List[Article]) -> List[Article]:
    seen = set()
    out: List[Article] = []
    for a in articles:
        key = sha1(normalize_text(a.company + " " + a.title))
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out


def infer_event_type(text: str) -> str:
    t = text.lower()
    mapping = [
        ("earnings", ["earnings", "guidance", "quarter", "q1", "q2", "q3", "q4", "revenue", "profit", "margin"]),
        ("product", ["launch", "release", "chip", "gpu", "copilot", "windows", "azure", "blackwell", "cuda"]),
        ("regulatory/legal", ["doj", "sec", "ftc", "lawsuit", "antitrust", "regulator", "ban", "export control"]),
        ("m&a", ["acquire", "acquisition", "merger", "buyout"]),
        ("security", ["breach", "hack", "vulnerability", "zero-day", "ransomware"]),
        ("partnership/customer", ["partnership", "deal", "customer", "contract", "orders", "supply"]),
        ("macro/market", ["rates", "inflation", "fed", "macro", "market selloff", "recession"]),
        ("management", ["ceo", "cfo", "resign", "appoint", "leadership"]),
    ]
    for label, keys in mapping:
        if any(k in t for k in keys):
            return label
    return "general"


def infer_importance(text: str, group: List[Article]) -> int:
    t = text.lower()
    score = 1

    if any(k in t for k in ["sec", "doj", "ftc", "lawsuit", "antitrust", "ban", "export control"]):
        score = max(score, 4)
    if any(k in t for k in ["earnings", "guidance", "revenue", "profit", "margin"]):
        score = max(score, 4)
    if any(k in t for k in ["acquire", "acquisition", "merger"]):
        score = max(score, 4)
    if any(k in t for k in ["launch", "release", "announce", "introduce"]):
        score = max(score, 3)

    # More sources → higher confidence/importance
    if len(group) >= 3:
        score = max(score, 3)
    if len(group) >= 6:
        score = max(score, 4)

    # trusted domains present
    if any(trust_score(a.link) == 2 for a in group):
        score = max(score, 3)

    return min(score, 5)


def cluster_by_keywords(articles: List[Article]) -> List[EventCluster]:
    """
    MVP clustering:
      - normalize title
      - strip tickers/company words
      - use signature = first 10 normalized tokens
    Production should use embeddings + semantic clustering.
    """
    buckets: Dict[str, List[Article]] = {}

    for a in articles:
        nt = normalize_text(a.title)
        nt = re.sub(r"\b(nvidia|nvda|microsoft|msft)\b", "", nt).strip()
        toks = nt.split()
        sig = " ".join(toks[:10]) if toks else nt
        sig = sig[:120]
        buckets.setdefault(a.company + "::" + sig, []).append(a)

    clusters: List[EventCluster] = []
    for k, group in buckets.items():
        company = group[0].company

        rep = sorted(group, key=lambda x: len(x.title), reverse=True)[0]

        snips: List[str] = []
        seen_snip = set()
        for g in sorted(
            group,
            key=lambda x: (x.published_utc or dt.datetime.min.replace(tzinfo=dt.timezone.utc)),
            reverse=True
        ):
            s = (g.snippet or "").strip()
            if s and s not in seen_snip:
                snips.append(s)
                seen_snip.add(s)

        canonical = " ".join(snips)[:900] if snips else ""

        event_type = infer_event_type(rep.title + " " + canonical)
        importance = infer_importance(rep.title + " " + canonical, group)
        confidence = 4 if len(group) >= 2 else 3
        event_id = sha1(k)[:10]

        clusters.append(EventCluster(
            company=company,
            event_id=event_id,
            headline=rep.title,
            articles=group,
            canonical_summary=canonical,
            event_type=event_type,
            importance=importance,
            confidence=confidence,
        ))

    def most_recent_utc(c: EventCluster) -> dt.datetime:
        ts = [a.published_utc for a in c.articles if a.published_utc]
        return max(ts) if ts else dt.datetime.min.replace(tzinfo=dt.timezone.utc)

    clusters.sort(key=lambda c: (c.importance, most_recent_utc(c)), reverse=True)
    return clusters


# -----------------------------
# LLM (optional) - Gemini
# -----------------------------

def call_gemini(system: str, user: str) -> str:
    """
    Returns markdown text (may be empty on block/filters/errors).
    """
    if not GEMINI_API_KEY:
        return ""

    try:
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=system,
        )

        response = model.generate_content(
            user,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 4096,
            },
        )

        # response.text can be empty in some edge cases
        text = getattr(response, "text", None)
        if text:
            return text

        # fallback: try candidates
        candidates = getattr(response, "candidates", None)
        if candidates:
            parts = []
            for cand in candidates:
                content = getattr(cand, "content", None)
                if not content:
                    continue
                for part in getattr(content, "parts", []) or []:
                    part_text = getattr(part, "text", None)
                    if part_text:
                        parts.append(part_text)
            if parts:
                return "\n".join(parts)

        return ""

    except Exception as e:
        return f"_Model call failed: {e}_"


def llm_analyze_clusters(clusters: List[EventCluster]) -> str:
    """
    Build a single daily analysis section, grouped by company.
    """
    if not GEMINI_API_KEY:
        return (
            "_LLM analysis disabled (GEMINI_API_KEY not set)._ \n\n"
            "You can still use this report for headlines + event clustering.\n"
        )

    facts: Dict[str, Any] = {}
    for c in clusters[:12]:
        facts.setdefault(c.company, []).append({
            "event_id": c.event_id,
            "headline": c.headline,
            "event_type": c.event_type,
            "importance": c.importance,
            "confidence": c.confidence,
            "canonical_summary": c.canonical_summary,
            "sources": [
                {
                    "title": a.title,
                    "source": a.source,
                    "time_local": to_local_time_str(a.published_utc),
                    "link": a.link,
                }
                for a in sorted(
                    c.articles,
                    key=lambda x: (x.published_utc or dt.datetime.min.replace(tzinfo=dt.timezone.utc)),
                    reverse=True
                )[:4]
            ],
        })

    system = (
        "You are a senior equity research analyst. "
        "Write a daily briefing for long-term investors. "
        "Strictly separate FACTS (from provided sources) vs INFERENCES. "
        "Do not invent numbers or events. If evidence is insufficient, say so. "
        "Include at least one counterpoint per company. "
        "When making claims, reference relevant event_id(s)."
    )

    user = (
        "Companies: NVDA (NVIDIA) and MSFT (Microsoft).\n"
        "Task: Generate a professional daily brief in Markdown.\n\n"
        "Output structure:\n"
        "## Executive Summary (5-8 bullets)\n"
        "## NVDA\n"
        "- Key Events (bullets with event_id)\n"
        "- Impact Analysis (Short / Medium / Long)\n"
        "- Expectation vs Surprise (what might be priced in)\n"
        "- Counterpoints & Risks\n"
        "- What to Monitor Next (concrete)\n"
        "## MSFT (same sections)\n"
        "## Appendix: Evidence (list event_id -> sources)\n\n"
        "Facts JSON (treat as evidence; do not modify):\n"
        "```json\n"
        f"{json.dumps(facts, ensure_ascii=False)}\n"
        "```\n"
    )

    out = call_gemini(system, user)
    return out.strip() if out else "_No analysis generated (empty model response)._"


# -----------------------------
# Report rendering
# -----------------------------

def render_md(clusters: List[EventCluster], analysis_md: str) -> str:
    today_local = dt.datetime.now(tz=LOCAL_TZ).strftime("%Y-%m-%d")
    lines: List[str] = []
    lines.append(f"# Daily News Brief — NVDA + MSFT — {today_local}\n")
    lines.append("> Generated by demo agent: RSS retrieval → dedup → light clustering → (optional) LLM analysis.\n")

    lines.append("## Top Event Clusters\n")
    for c in clusters[:15]:
        lines.append(f"### [{c.company}] ({c.event_type}) {c.headline}")
        lines.append(f"- event_id: `{c.event_id}` | importance: {c.importance}/5 | confidence: {c.confidence}/5")
        if c.canonical_summary:
            lines.append(f"- summary: {c.canonical_summary}")
        lines.append("- sources:")
        for a in sorted(
            c.articles,
            key=lambda x: (x.published_utc or dt.datetime.min.replace(tzinfo=dt.timezone.utc)),
            reverse=True
        )[:5]:
            lines.append(f"  - {to_local_time_str(a.published_utc)} — {a.source} — [{a.title}]({a.link})")
        lines.append("")

    lines.append("\n---\n")
    lines.append("## Model Analysis\n")
    lines.append(analysis_md.strip() if analysis_md.strip() else "_No analysis generated._")

    return "\n".join(lines)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ensure_dirs()

    all_articles: List[Article] = []
    print("Fetching RSS...")
    for c in tqdm(COMPANIES):
        try:
            items = fetch_rss_articles(c)
            all_articles.extend(items)
        except Exception as e:
            print(f"[WARN] Failed to fetch {c['ticker']}: {e}")

    all_articles = simple_dedup(all_articles)

    all_articles.sort(
        key=lambda a: (a.published_utc or dt.datetime.min.replace(tzinfo=dt.timezone.utc)),
        reverse=True
    )

    clusters = cluster_by_keywords(all_articles)

    analysis_md = llm_analyze_clusters(clusters)

    report_md = render_md(clusters, analysis_md)

    today_local = dt.datetime.now(tz=LOCAL_TZ).strftime("%Y-%m-%d")
    out_path = os.path.join(OUT_DIR, f"{today_local}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\n✅ Report written to: {out_path}")


if __name__ == "__main__":
    main()