#!/usr/bin/env python3
"""
Fixed focused crawler for SHL product catalog.

Usage (PowerShell or cmd):
  python crawler.py --start-url "https://www.shl.com/products/product-catalog/" --out-dir ./data --max-pages 500

Outputs:
  ./data/catalog.jsonl
  ./data/catalog.csv
  ./data/raw_html/*.html

Notes:
 - Installs: pip install -r requirements.txt
 - Optional Playwright for JS rendering: pip install playwright ; playwright install
"""

import argparse
import json
import logging
import os
import random
import re
import time
from collections import deque
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Try to import Playwright fallback (lazy)
USE_PLAYWRIGHT = True
try:
    if USE_PLAYWRIGHT:
        from playwright.sync_api import sync_playwright
except Exception:
    sync_playwright = None

# ---------------- Config ----------------
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SHL-Product-Crawler/3.0)"}
REQUEST_TIMEOUT = 20
RETRY_COUNT = 2
MIN_DESC_LEN = 140
SLEEP_RANGE = (0.4, 1.1)
DEFAULT_LINK_FILTER = r"/products/product-catalog/"
# ----------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def check_robots_allowed(start_url: str) -> bool:
    """Very small robots.txt check for disallow rules. Conservative: allow if unknown."""
    try:
        parsed = urlparse(start_url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        r = requests.get(robots_url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return True
        lines = [ln.strip() for ln in r.text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        ua = None
        disallows = []
        for ln in lines:
            if ln.lower().startswith("user-agent:"):
                ua = ln.split(":", 1)[1].strip()
            elif ln.lower().startswith("disallow:"):
                path = ln.split(":", 1)[1].strip()
                if ua == "*" or ua is None:
                    disallows.append(path)
        # if full site disallowed, block
        for d in disallows:
            if d == "/":
                return False
        # otherwise allow
        return True
    except Exception as e:
        logging.warning("robots.txt check failed (%s) — proceeding by default", e)
        return True


def fetch_requests(url: str) -> str:
    """Fetch using requests with retries. Returns HTML text."""
    last_exc = None
    for attempt in range(RETRY_COUNT + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_exc = e
            time.sleep(0.6 + attempt * 0.6)
    raise RuntimeError(f"requests fetch failed for {url}: {last_exc}")


def fetch_playwright(url: str, timeout: int = 35000) -> str:
    """Render page with Playwright (synchronous). Returns page HTML content."""
    if sync_playwright is None:
        raise RuntimeError("Playwright not installed. Install with `pip install playwright` and run `playwright install`.")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(url, timeout=timeout, wait_until="load")
            # small wait for dynamic content
            page.wait_for_timeout(900)
            return page.content()
        finally:
            browser.close()


def extract_metadata(html: str, url: str) -> dict:
    """Extract title, description, tags, and heuristic test_type."""
    soup = BeautifulSoup(html, "html.parser")
    title = ""
    if soup.select_one("h1"):
        title = soup.select_one("h1").get_text(strip=True)
    elif soup.title:
        title = soup.title.get_text(strip=True)

    # Prefer content inside <main> or article
    main = soup.find("main") or soup.find("article") or soup
    paragraphs = [p.get_text(" ", strip=True) for p in main.select("p") if p.get_text(strip=True)]
    description = "\n\n".join(paragraphs).strip()
    if not description:
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            description = meta_desc["content"].strip()

    # tags
    tags = []
    for sel in ["ul.tags li", ".tags li", ".product-tags li", ".key-terms li", ".taxonomy a"]:
        for el in soup.select(sel):
            t = el.get_text(strip=True)
            if t:
                tags.append(t)
    if not tags:
        meta_kw = soup.find("meta", attrs={"name": "keywords"})
        if meta_kw and meta_kw.get("content"):
            tags = [k.strip() for k in meta_kw["content"].split(",") if k.strip()]

    # test_type heuristics
    blob = (title + " " + description + " " + " ".join(tags)).lower()
    test_type = ""
    if re.search(r"\b(knowledge|aptitude|verbal|numerical|logical|cognitive|technical|coding)\b", blob):
        test_type = "K"
    if re.search(r"\b(personality|behaviour|behavioral|situational|soft skill|competenc)\b", blob):
        test_type = ("K+P" if test_type == "K" else "P")

    return {
        "assessment_name": title,
        "url": url.rstrip("/"),
        "description": description,
        "tags": tags,
        "test_type": test_type,
    }


def find_links_in_html(html: str, base_url: str, link_filter: str) -> list:
    """Return list of normalized links (same domain) matching the provided regex filter."""
    soup = BeautifulSoup(html, "html.parser")
    out = set()
    base_host = urlparse(base_url).netloc
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("javascript:", "mailto:", "#")):
            continue
        full = urljoin(base_url, href)
        p = urlparse(full)
        if p.netloc != base_host:
            continue
        norm = f"{p.scheme}://{p.netloc}{p.path}"
        if re.search(link_filter, norm):
            out.add(norm.rstrip("/"))
    return list(out)


def save_raw_html(out_dir: str, url: str, html: str, suffix: str = ""):
    parsed = urlparse(url)
    safe_path = (parsed.path or "/").strip("/").replace("/", "_") or "index"
    os.makedirs(os.path.join(out_dir, "raw_html"), exist_ok=True)
    fname = f"{parsed.netloc}_{safe_path}{suffix}.html"
    with open(os.path.join(out_dir, "raw_html", fname), "w", encoding="utf-8") as f:
        f.write(html)


def write_jsonl(path: str, items: list):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def write_csv(path: str, items: list):
    import csv
    keys = ["assessment_name", "url", "test_type", "tags", "description"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for it in items:
            w.writerow({
                "assessment_name": it.get("assessment_name", ""),
                "url": it.get("url", ""),
                "test_type": it.get("test_type", ""),
                "tags": ";".join(it.get("tags", [])),
                "description": (it.get("description") or "").replace("\n", " ")
            })


def crawl(start_urls: list, link_filter: str, max_pages: int, out_dir: str) -> list:
    if not check_robots_allowed(start_urls[0]):
        logging.error("Crawling disallowed by robots.txt for this domain. Exiting.")
        return []

    seen = set()
    q = deque(start_urls)
    results = []
    pbar = tqdm(total=max_pages, desc="Collected")

    while q and len(results) < max_pages:
        url = q.popleft()
        if url in seen:
            continue
        seen.add(url)

        # 1) Try requests
        html = None
        try:
            html = fetch_requests(url)
        except Exception as e:
            logging.debug("Requests fetch failed for %s: %s", url, e)

        # 2) Extract metadata from requests HTML (if any)
        meta = extract_metadata(html or "", url) if html else {"description": ""}

        # 3) If page likely JS-driven (short desc) or it's the catalog root, render with Playwright
        is_catalog_root = bool(re.search(r"/products/product-catalog/?$", url))
        use_rendered_html = None
        if (not html) or len(meta.get("description", "") or "") < MIN_DESC_LEN or is_catalog_root:
            if sync_playwright:
                try:
                    logging.info("Rendering with Playwright: %s", url)
                    rendered = fetch_playwright(url)
                    use_rendered_html = rendered
                    meta = extract_metadata(rendered, url)
                    save_raw_html(out_dir, url, rendered, "_rendered")
                except Exception as e:
                    logging.warning("Playwright render failed for %s: %s", url, e)
            else:
                logging.debug("Playwright not available and rendering needed for %s", url)

        # Save the raw requests HTML copy if present
        if html:
            save_raw_html(out_dir, url, html, "")

        # Heuristic: product pages we're interested in mostly have '/products/product-catalog/view/' in path
        path = urlparse(url).path.lower()
        if re.search(r"/products/product-catalog/(view/)?", path) and (meta.get("assessment_name") or len(meta.get("description", "")) > 40):
            results.append(meta)
            pbar.update(1)

        # Discover links from the rendered DOM if available, otherwise from requests DOM
        dom_for_links = use_rendered_html if use_rendered_html is not None else (html or "")
        discovered = find_links_in_html(dom_for_links, url, link_filter)
        for d in discovered:
            if d not in seen and d not in q:
                q.append(d)

        time.sleep(random.uniform(*SLEEP_RANGE))

    pbar.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-url", required=True, help="Comma-separated start URL(s). Default target: https://www.shl.com/products/product-catalog/")
    parser.add_argument("--link-filter", default=DEFAULT_LINK_FILTER, help="Regex for links to follow (default targets /products/product-catalog/)")
    parser.add_argument("--max-pages", type=int, default=500, help="Maximum product pages to collect")
    parser.add_argument("--out-dir", default="./data", help="Output directory")
    args = parser.parse_args()

    start_urls = [u.strip().rstrip("/") for u in args.start_url.split(",") if u.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    logging.info("Starting crawl from: %s", start_urls)
    items = crawl(start_urls, args.link_filter, args.max_pages, args.out_dir)

    jsonl_path = os.path.join(args.out_dir, "catalog.jsonl")
    csv_path = os.path.join(args.out_dir, "catalog.csv")
    write_jsonl(jsonl_path, items)
    write_csv(csv_path, items)
    logging.info("Crawl complete. Saved %d items -> %s, %s", len(items), jsonl_path, csv_path)


if __name__ == "__main__":
    main()
