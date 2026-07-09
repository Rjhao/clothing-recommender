"""Crawl real clothing photos from Pexels by category, save URLs to JSON.

Run:  PEXELS_API_KEY=xxx python -m data.crawl_pexels_images
Output: data/pexels_images.json   {category: [url, ...]}

These URLs replace the loremflickr fallback in kg.image_url.resolve_image so
every clothing card shows a real, category-matched Pexels stock photo.
Pexels CDN (images.pexels.com) is reachable where wikimedia.org is not.
"""
import json
import logging
import os
import sys
import urllib.parse
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

API = "https://api.pexels.com/v1/search"

# Category -> Pexels search keyword (real, clean stock photos).
CATEGORY_KEYWORD = {
    "top": "t-shirt",
    "bottom": "jeans",
    "dress": "dress",
    "outerwear": "jacket",
    "shoes": "shoes",
    "accessories": "handbag",
}
PER_PAGE = 80  # Pexels max per page; one request per category
OUT = os.path.join(os.path.dirname(__file__), "pexels_images.json")


def fetch_photos(query: str, per_page: int, key: str) -> dict:
    url = f"{API}?query={urllib.parse.quote(query)}&per_page={per_page}&orientation=portrait"
    req = urllib.request.Request(url, headers={
        "Authorization": key,
        "User-Agent": "clothing-recommender/1.0",
    })
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    key = os.environ.get("PEXELS_API_KEY")
    if not key:
        log.error("PEXELS_API_KEY env var not set")
        sys.exit(1)
    result = {}
    for cat, kw in CATEGORY_KEYWORD.items():
        try:
            data = fetch_photos(kw, PER_PAGE, key)
            urls = [
                p["src"]["large"]
                for p in data.get("photos", [])
                if p.get("src", {}).get("large")
            ]
            result[cat] = urls
            log.info("%s (%s): %d photos", cat, kw, len(urls))
        except Exception as e:
            log.error("%s (%s) failed: %s", cat, kw, e)
            result[cat] = []
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log.info("Saved to %s: %s", OUT, {k: len(v) for k, v in result.items()})


if __name__ == "__main__":
    main()
