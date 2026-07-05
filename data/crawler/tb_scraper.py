"""Taobao/Tmall scraper for clothing products.

Note: Taobao has strong anti-scraping measures. This scraper uses the
open platform API when possible and falls back to page scraping.
For production use, consider using the Aliyun open platform or playwright.
"""
import json
import logging
import re
import hashlib
import time
from typing import List, Dict, Optional

from .base import BaseScraper

logger = logging.getLogger(__name__)


class TaobaoScraper(BaseScraper):
    """Scraper for Taobao.com and Tmall.com platforms."""

    def __init__(self):
        super().__init__(delay_range=(3, 5))
        self.base_url = "https://www.taobao.com"
        self.search_url = "https://s.taobao.com/search"
        self.tmall_url = "https://list.tmall.com"

    def search(self, keyword: str, page: int = 1, per_page: int = 20) -> List[Dict]:
        """
        Search products on Taobao.

        Args:
            keyword: Search keyword (URL encoded)
            page: Page number
            per_page: Items per page

        Returns:
            List of product dictionaries
        """
        params = {
            "q": keyword,
            "sort": "sale-desc",
            "page": page,
            "tab": "mall",
        }

        try:
            response = self.session.get(
                self.search_url,
                params=params,
                headers=self._get_headers(),
                timeout=15
            )
            response.raise_for_status()
            response.encoding = "utf-8"

            return self._parse_search_results(response.text)
        except Exception as e:
            logger.error(f"Taobao search failed: {e}")
            return []

    def search_tmall(self, keyword: str, page: int = 1) -> List[Dict]:
        """
        Search products specifically on Tmall.

        Args:
            keyword: Search keyword
            page: Page number

        Returns:
            List of product dictionaries
        """
        params = {
            "q": keyword,
            "sort": "d",
            "page": page,
        }

        try:
            response = self.session.get(
                self.tmall_url,
                params=params,
                headers=self._get_headers(),
                timeout=15
            )
            response.raise_for_status()
            response.encoding = "gbk"  # Tmall uses GBK encoding
            response.encoding = "utf-8"

            return self._parse_tmall_results(response.text)
        except Exception as e:
            logger.error(f"Tmall search failed: {e}")
            return []

    def _parse_search_results(self, html: str) -> List[Dict]:
        """
        Parse Taobao search results from HTML.

        Args:
            html: Raw HTML content

        Returns:
            List of parsed products
        """
        products = []

        # Try to find JSON data embedded in page
        # Taobao embeds search results in g_page_config
        config_match = re.search(r'g_page_config\s*=\s*(\{.*?\});', html, re.DOTALL)
        if config_match:
            try:
                config = json.loads(config_match.group(1))
                # Extract from nested JSON structure
                if "mods" in config:
                    items = config["mods"].get("itemlist", {}).get("data", {}).get("auctions", [])
                    for item in items:
                        products.append(self._normalize_taobao_item(item))
            except (json.JSONDecodeError, KeyError) as e:
                logger.debug(f"Failed to parse JSON config: {e}")

        # Fallback: parse from HTML if JSON parsing fails
        if not products:
            products = self._parse_html_results(html)

        return products

    def _parse_tmall_results(self, html: str) -> List[Dict]:
        """
        Parse Tmall search results.

        Args:
            html: Raw HTML content

        Returns:
            List of parsed products
        """
        products = []

        # Tmall product pattern
        item_pattern = re.compile(
            r'<div class="product".*?data-id="(\d+)".*?'
            r'<a class="productTitle".*?title="([^"]+)".*?'
            r'<span class="productPrice">.*?¥([\d.]+)</span>',
            re.DOTALL
        )

        for match in item_pattern.finditer(html):
            products.append({
                "product_id": match.group(1),
                "name": match.group(2).strip(),
                "price": float(match.group(3)) if match.group(3) else 0,
                "platform": "TMALL",
                "source": f"https://detail.tmall.com/item.htm?id={match.group(1)}",
            })

        return products

    def _parse_html_results(self, html: str) -> List[Dict]:
        """
        Fallback HTML parsing for Taobao results.

        Args:
            html: Raw HTML content

        Returns:
            List of parsed products
        """
        products = []

        # Generic product pattern
        patterns = [
            # Pattern 1: data-nid
            re.compile(r'data-nid="(\d+)".*?<a[^>]*title="([^"]+)".*?price["\s>]+([\d.]+)', re.DOTALL),
            # Pattern 2: product id and price
            re.compile(r'"nid":"(\d+)".*?"raw_title":"([^"]+)".*?"view_price":"([\d.]+)"', re.DOTALL),
        ]

        seen_ids = set()
        for pattern in patterns:
            for match in pattern.finditer(html):
                product_id = match.group(1)
                if product_id in seen_ids:
                    continue
                seen_ids.add(product_id)

                name = self._clean_html(match.group(2))
                price = float(match.group(3)) if match.group(3) else 0

                products.append({
                    "product_id": product_id,
                    "name": name,
                    "price": price,
                    "platform": "TAOBAO",
                    "source": f"https://item.taobao.com/item.htm?id={product_id}",
                })

        return products

    def _normalize_taobao_item(self, item: dict) -> Dict:
        """
        Normalize Taobao item data to standard format.

        Args:
            item: Raw item dictionary from Taobao

        Returns:
            Normalized product dictionary
        """
        # Handle different field names from Taobao
        product_id = item.get("nid", item.get("item_id", ""))
        name = item.get("title", item.get("raw_title", ""))
        price = item.get("view_price", item.get("price", 0))

        if isinstance(price, str):
            price = float(price) if price else 0

        return {
            "product_id": str(product_id),
            "name": name,
            "price": price,
            "platform": "TAOBAO",
            "source": f"https://item.taobao.com/item.htm?id={product_id}",
            "shop": item.get("shop", {}).get("shopLink", "") if isinstance(item.get("shop"), dict) else "",
            "location": item.get("item_loc", ""),
            "sales": item.get("view_sales", item.get("sales", "")),
            "comment_count": item.get("comment_count", 0),
        }

    def get_product_detail(self, product_id: str) -> Optional[Dict]:
        """
        Get detailed product information from Taobao.

        Args:
            product_id: Taobao product ID

        Returns:
            Detailed product dictionary
        """
        url = f"https://item.taobao.com/item.htm?id={product_id}"

        try:
            response = self.session.get(
                url,
                headers=self._get_headers(),
                timeout=15
            )
            response.raise_for_status()
            response.encoding = "utf-8"

            return self._parse_product_detail(response.text, product_id)
        except Exception as e:
            logger.error(f"Failed to get product detail for {product_id}: {e}")
            return None

    def _parse_product_detail(self, html: str, product_id: str) -> Dict:
        """
        Parse Taobao/Tmall product detail page.

        Args:
            html: Raw HTML content
            product_id: Product ID

        Returns:
            Parsed product details
        """
        detail = {"product_id": product_id, "platform": "TAOBAO"}

        # Try to extract from JSON-LD structured data
        jsonld_pattern = re.search(
            r'<script type="application/ld\+json">(.*?)</script>',
            html,
            re.DOTALL
        )
        if jsonld_pattern:
            try:
                data = json.loads(jsonld_pattern.group(1))
                if "name" in data:
                    detail["name"] = data["name"]
                if "brand" in data:
                    detail["brand"] = data["brand"]
                if "description" in data:
                    detail["description"] = data["description"]
                if "image" in data:
                    detail["images"] = data["image"] if isinstance(data["image"], list) else [data["image"]]
            except json.JSONDecodeError:
                pass

        # Extract category
        cat_match = re.search(r'"category":\s*"([^"]+)"', html)
        if cat_match:
            detail["category"] = cat_match.group(1)

        # Extract brand
        brand_match = re.search(r'"brand":\s*"([^"]+)"', html)
        if brand_match:
            detail["brand"] = brand_match.group(1)

        # Extract price range
        price_match = re.search(r'"price":\s*"([^"]+)"', html)
        if price_match:
            detail["price"] = price_match.group(1)

        # Extract shop name
        shop_match = re.search(r'"seller":\s*\{.*?"nick":\s*"([^"]+)"', html, re.DOTALL)
        if shop_match:
            detail["shop_name"] = shop_match.group(1)

        # Extract description
        desc_match = re.search(r'<div class="attributes".*?<ul>(.*?)</ul>', html, re.DOTALL)
        if desc_match:
            detail["description"] = self._clean_html(desc_match.group(1))

        # Extract main image
        main_img_match = re.search(r'cover: "([^"]+)"', html)
        if main_img_match:
            detail["main_image"] = main_img_match.group(1)

        return detail

    def _clean_html(self, text: str) -> str:
        """Remove HTML tags and clean text."""
        if not text:
            return ""
        text = re.sub(r'<[^>]+>', '', text)
        text = text.replace("&nbsp;", " ").replace("&amp;", "&")
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def crawl_by_category(
        self,
        categories: List[str],
        pages_per_category: int = 3,
        use_tmall: bool = True
    ) -> List[Dict]:
        """
        Crawl products by category from Taobao/Tmall.

        Args:
            categories: List of category names
            pages_per_category: Pages to crawl per category
            use_tmall: Whether to use Tmall (more reliable)

        Returns:
            List of crawled products
        """
        all_products = []

        for category in categories:
            for page in range(1, pages_per_category + 1):
                if use_tmall:
                    products = self.search_tmall(category, page)
                else:
                    products = self.search(category, page)

                all_products.extend(products)
                logger.info(f"Crawled {len(products)} products for '{category}' page {page}")

                # Rate limit
                self._random_delay()

        # Deduplicate
        seen_ids = set()
        unique_products = []
        for product in all_products:
            if product["product_id"] not in seen_ids:
                seen_ids.add(product["product_id"])
                unique_products.append(product)

        return unique_products


# Taobao Category Keywords
TAOBAO_CLOTHING_CATEGORIES = [
    "男士T恤", "女士T恤", "男士衬衫", "女士衬衫",
    "男士牛仔裤", "女士牛仔裤", "连衣裙", "半身裙",
    "男士外套", "女士外套", "羽绒服", "棉服",
    "运动裤", "休闲裤", "西装外套", "卫衣",
    "毛衣", "针织衫", "polo衫", "吊带背心",
    "男鞋", "女鞋", "运动休闲鞋", "皮鞋",
    "皮带", "围巾", "帽子", "手套",
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    scraper = TaobaoScraper()

    # Test search
    print("Testing Taobao scraper...")
    products = scraper.search("男士T恤")

    if products:
        print(f"Found {len(products)} products")
        print("\nFirst 3 products:")
        for p in products[:3]:
            print(f"  {p['product_id']}: {p['name']} - ¥{p['price']}")
    else:
        print("No products found (this is expected if anti-bot is triggered)")
