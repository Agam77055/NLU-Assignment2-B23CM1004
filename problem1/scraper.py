"""
scraper.py — Crawls IITJ website pages and saves raw text

BFS-based crawler that stays within iitj.ac.in domain.
Seeds from department/research/faculty/academic-regulation pages.

Usage:
    python scraper.py

Outputs raw page texts to data/raw/ directory.

Author: Agam Harpreet Singh (B23CM1004)
"""

import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# ============================================================
# CONFIG
# ============================================================

SEED = 42  # not used here really but keeping it for uniformity across files

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

os.makedirs(DATA_DIR, exist_ok=True)

# TODO: maybe add more URLs later — placement and alumni pages could be useful
SEED_URLS = [
    "https://iitj.ac.in/",
    "https://iitj.ac.in/academics/index.php?id=phd",
    "https://iitj.ac.in/academics/index.php?id=btech",
    "https://iitj.ac.in/academics/index.php?id=mtech",
    "https://iitj.ac.in/department/index.php?id=cse",
    "https://iitj.ac.in/department/index.php?id=ee",
    "https://iitj.ac.in/department/index.php?id=me",
    "https://iitj.ac.in/research/index.php?id=research_highlights",
    "https://iitj.ac.in/academics/index.php?id=academic_regulations",
    "https://iitj.ac.in/people/index.php?id=faculty_list",
    "https://iitj.ac.in/academics/index.php?id=hostel",
    "https://iitj.ac.in/placement/index.php?id=placement",
]


# ============================================================
# SCRAPER CLASS
# ============================================================

class IITJScraper:
    def __init__(self, max_pages=100, delay=0.5):
        self.max_pages = max_pages
        self.delay = delay  # be polite to the server
        self.visited = set()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; IITJResearchBot/1.0)"
        })

    def _fetch_page(self, url):
        """Try to get page content, return None if it fails"""
        try:
            resp = self.session.get(url, timeout=10)
            if resp.status_code == 200:
                return resp.text
        except Exception as e:
            print(f"    failed to fetch {url}: {e}")
        return None

    def _is_valid_url(self, url):
        """Check if we should crawl this URL"""
        parsed = urlparse(url)
        # only stay within iitj.ac.in
        if "iitj.ac.in" not in parsed.netloc:
            return False
        # skip non-text resources
        skip_exts = ['.pdf', '.jpg', '.png', '.gif', '.zip', '.doc', '.ppt']
        if any(url.lower().endswith(ext) for ext in skip_exts):
            return False
        return True

    def _extract_links(self, html, base_url):
        """Pull all valid links from a page"""
        soup = BeautifulSoup(html, "html.parser")
        links = []
        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            full = urljoin(base_url, href)
            # normalize - drop fragments and query strings mostly
            full = full.split("#")[0].strip()
            if full and self._is_valid_url(full) and full not in self.visited:
                links.append(full)
        return links

    def _is_english(self, text):
        """Rough check - if >70% of alpha chars are ASCII, call it English"""
        if not text:
            return False
        alpha_chars = [c for c in text if c.isalpha()]
        if len(alpha_chars) < 50:  # too short to be useful
            return False
        ascii_count = sum(1 for c in alpha_chars if ord(c) < 128)
        return (ascii_count / len(alpha_chars)) > 0.70

    def _extract_text(self, html):
        """Extract meaningful text from HTML, skip nav/footer/scripts"""
        soup = BeautifulSoup(html, "html.parser")

        # remove things we don't want
        for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # grab text from content tags
        content_tags = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "article"])

        pieces = []
        for tag in content_tags:
            txt = tag.get_text(separator=" ", strip=True)
            if len(txt) > 30:  # skip tiny fragments
                pieces.append(txt)

        return "\n".join(pieces)

    def crawl(self, seed_urls):
        """BFS crawler - visits pages breadth-first from seed URLs"""
        queue = list(seed_urls)
        pages = {}  # url -> text

        print(f"Starting crawl from {len(seed_urls)} seed URLs...")
        print(f"Target: {self.max_pages} pages")

        while queue and len(pages) < self.max_pages:
            url = queue.pop(0)

            if url in self.visited:
                continue
            self.visited.add(url)

            print(f"  [{len(pages)+1}/{self.max_pages}] {url[:80]}")

            html = self._fetch_page(url)
            if html is None:
                continue

            text = self._extract_text(html)

            if self._is_english(text):
                pages[url] = text
                # add new links to the queue
                new_links = self._extract_links(html, url)
                queue.extend(new_links[:15])  # don't grab too many from one page

            time.sleep(self.delay)

        print(f"\nCrawl done. Got {len(pages)} pages.")
        return pages

    def save_raw(self, pages, raw_dir):
        """Save each page to a numbered text file"""
        os.makedirs(raw_dir, exist_ok=True)
        for i, (url, text) in enumerate(pages.items()):
            fpath = os.path.join(raw_dir, f"{i}.txt")
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(f"SOURCE: {url}\n\n")
                f.write(text)
        print(f"Saved {len(pages)} raw pages to {raw_dir}")


# ============================================================
# MAIN
# ============================================================

def main():
    raw_dir = os.path.join(DATA_DIR, "raw")

    scraper = IITJScraper(max_pages=100, delay=0.5)
    pages = scraper.crawl(SEED_URLS)

    if not pages:
        print("ERROR: no pages collected. Check network / URLs.")
        return

    scraper.save_raw(pages, raw_dir)

    # quick stats
    total_chars = sum(len(t) for t in pages.values())
    print(f"\nQuick stats:")
    print(f"  Pages collected : {len(pages)}")
    print(f"  Total chars     : {total_chars:,}")
    print(f"  Avg page chars  : {total_chars // len(pages):,}")


if __name__ == "__main__":
    main()
