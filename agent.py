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

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

import markdown as md_lib


# -----------------------------
# Config
# -----------------------------

LOCAL_TZ = tz.gettz("America/Los_Angeles")

COMPANIES = [
    {"ticker": "NVDA", "name": "NVIDIA", "query": "NVIDIA OR NVDA"},
    {"ticker": "MSFT", "name": "Microsoft", "query": "Microsoft OR MSFT"},
]

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"

OUT_DIR = "reports"
CACHE_DIR = "cache"
MAX_ITEMS_PER_COMPANY = 30

TRUSTED_DOMAIN_HINTS = [
    "reuters.com", "bloomberg.com", "wsj.com", "ft.com", "nytimes.com",
    "theverge.com", "techcrunch.com", "arstechnica.com",
    "sec.gov", "microsoft.com", "nvidia.com", "investor", "ir.",
]

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Email (SMTP)
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")

EMAIL_FROM = os.getenv("EMAIL_FROM", SMTP_USER)
EMAIL_TO = os.getenv("EMAIL_TO", "")
EMAIL_CC = os.getenv("EMAIL_CC", "")
EMAIL_BCC = os.getenv("EMAIL_BCC", "")
EMAIL_SUBJECT_PREFIX = os.getenv("EMAIL_SUBJECT_PREFIX", "[Daily Brief]")

# Schedule
SCHEDULE_HOUR = int(os.getenv("SCHEDULE_HOUR", "9"))
SCHEDULE_MINUTE = int(os.getenv("SCHEDULE_MINUTE", "0"))


# -----------------------------
# Data model
# -----------------------------

@dataclass
class Article:
    company: str
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
    importance: int
    confidence: int


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


def today_local_str() -> str:
    return dt.datetime.now(tz=LOCAL_TZ).strftime("%Y-%m-%d")


def parse_emails(csv: str) -> List[str]:
    if not csv.strip():
        return []
    parts = [p.strip() for p in csv.split(",")]
    return [p for p in parts if p]


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

    if len(group) >= 3:
        score = max(score, 3)
    if len(group) >= 6:
        score = max(score, 4)

    if any(trust_score(a.link) == 2 for a in group):
        score = max(score, 3)

    return min(score, 5)


def cluster_by_keywords(articles: List[Article]) -> List[EventCluster]:
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
# LLM - Gemini
# -----------------------------

def call_gemini(system: str, user: str) -> str:
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

        text = getattr(response, "text", None)
        if text:
            return text

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
    today = today_local_str()
    lines: List[str] = []
    lines.append(f"# Daily News Brief — NVDA + MSFT — {today}\n")
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


def md_to_html(markdown_text: str) -> str:
    """
    Convert Markdown to email-friendly HTML.
    """
    html_body = md_lib.markdown(
        markdown_text,
        extensions=["extra", "sane_lists", "tables", "toc"],
        output_format="html5",
    )

    # Very simple inline CSS for email readability
    style = """
    <style>
      body { font-family: Arial, sans-serif; line-height: 1.4; }
      h1,h2,h3 { margin: 16px 0 8px; }
      code, pre { background: #f6f8fa; padding: 2px 4px; border-radius: 4px; }
      pre { padding: 12px; overflow-x: auto; }
      a { color: #1a0dab; }
      table { border-collapse: collapse; }
      th, td { border: 1px solid #ddd; padding: 6px 8px; }
    </style>
    """
    return f"<!doctype html><html><head>{style}</head><body>{html_body}</body></html>"


# -----------------------------
# Email sending
# -----------------------------

def validate_email_config() -> None:
    missing = []
    if not SMTP_HOST:
        missing.append("SMTP_HOST")
    if not SMTP_PORT:
        missing.append("SMTP_PORT")
    if not SMTP_USER:
        missing.append("SMTP_USER")
    if not SMTP_PASS:
        missing.append("SMTP_PASS")
    if not EMAIL_TO.strip():
        missing.append("EMAIL_TO")
    if missing:
        raise RuntimeError(f"Missing email config env vars: {', '.join(missing)}")


def send_email_report(subject: str, markdown_text: str, md_path: str) -> None:
    validate_email_config()

    to_list = parse_emails(EMAIL_TO)
    cc_list = parse_emails(EMAIL_CC)
    bcc_list = parse_emails(EMAIL_BCC)

    # Email container
    msg = MIMEMultipart("mixed")
    msg["From"] = EMAIL_FROM or SMTP_USER
    msg["To"] = ", ".join(to_list)
    if cc_list:
        msg["Cc"] = ", ".join(cc_list)
    msg["Subject"] = subject

    # Alternative part: plain text + HTML
    alt = MIMEMultipart("alternative")
    plain_text = (
        "Daily report is attached as Markdown.\n\n"
        "If your email client supports HTML, you should see a formatted version.\n"
    )
    alt.attach(MIMEText(plain_text, "plain", "utf-8"))

    html = md_to_html(markdown_text)
    alt.attach(MIMEText(html, "html", "utf-8"))

    msg.attach(alt)

    # Attach markdown file
    with open(md_path, "rb") as f:
        part = MIMEApplication(f.read(), _subtype="markdown")
    filename = os.path.basename(md_path)
    part.add_header("Content-Disposition", "attachment", filename=filename)
    msg.attach(part)

    # Send
    recipients = to_list + cc_list + bcc_list
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, recipients, msg.as_string())


# -----------------------------
# Pipeline
# -----------------------------

def generate_report() -> str:
    """
    Returns the path to the generated .md file.
    """
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

    today = today_local_str()
    out_path = os.path.join(OUT_DIR, f"{today}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"✅ Report written to: {out_path}")
    return out_path


def run_once_send_email() -> None:
    """
    Generate report and email it.
    """
    md_path = generate_report()
    with open(md_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    subject = f"{EMAIL_SUBJECT_PREFIX} NVDA+MSFT — {today_local_str()}"
    send_email_report(subject=subject, markdown_text=markdown_text, md_path=md_path)
    print("✅ Email sent.")


# -----------------------------
# Scheduler
# -----------------------------

def start_scheduler() -> None:
    """
    Runs forever and triggers at 09:00 America/Los_Angeles daily.
    """
    scheduler = BlockingScheduler(timezone="America/Los_Angeles")

    trigger = CronTrigger(hour=SCHEDULE_HOUR, minute=SCHEDULE_MINUTE)
    scheduler.add_job(
        run_once_send_email,
        trigger=trigger,
        id="daily_email_report",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )

    now = dt.datetime.now(tz=LOCAL_TZ).strftime("%Y-%m-%d %H:%M %Z")
    print(f"Scheduler started at {now}. Will send daily at {SCHEDULE_HOUR:02d}:{SCHEDULE_MINUTE:02d} {LOCAL_TZ.tzname(dt.datetime.now(tz=LOCAL_TZ))}.")
    scheduler.start()


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    # Two modes:
    # 1) Run once now:
    #    python agent.py --once
    # 2) Run scheduler (default):
    #    python agent.py
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        run_once_send_email()
    else:
        start_scheduler()