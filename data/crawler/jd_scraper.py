"""JD.com (京东) scraper for clothing products."""
import json
import logging
import re
from typing import List, Dict, Optional

from .base import BaseScraper

logger = logging.getLogger(__name__)


class JDScraper(BaseScraper):
    """Scraper for JD.com e-commerce platform."""

    def __init__(self):
        super().__init__(delay_range=(2, 4))
        self.base_url = "https://www.jd.com"
        self.search_url = "https://search.jd.com/Search"
        self.product_url = "https://item.jd.com/{pid}.html"

    def search(self, keyword: str, page: int = 1, per_page: int = 20) -> List[Dict]:
        """
        Search products on JD.com.

        Args:
            keyword: Search keyword (URL encoded)
            page: Page number
            per_page: Items per page

        Returns:
            List of product dictionaries
        """
        params = {
            "keyword": keyword,
            "enc": "utf-8",
            "page": page,
            "qrst": "1",
            "spm": "a0210.搜索.0.0",
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

            return self._parse_search_results(response.text, page)
        except Exception as e:
            logger.error(f"JD search failed: {e}")
            return []

    def _parse_search_results(self, html: str, page: int) -> List[Dict]:
        """
        Parse JD search results from HTML.

        Args:
            html: Raw HTML content
            page: Page number

        Returns:
            List of parsed products
        """
        products = []

        # JD uses lazy loading, try to extract from JSON data
        # Pattern: var searchParams = {...}
        params_match = re.search(r'var searchParams\s*=\s*(\{.*?\});', html, re.DOTALL)
        if params_match:
            try:
                search_data = json.loads(params_match.group(1))
                logger.debug(f"Found search params: {search_data}")
            except json.JSONDecodeError:
                pass

        # Extract product list from page
        # JD product items have class "gl-item"
        item_pattern = re.compile(
            r'<li class="gl-item".*?data-sku="(\d+)".*?'
            r'<div class="p-name p-name-type-2">.*?<em>(.*?)</em>.*?'
            r'<div class="p-price".*?<i>¥([\d.]+)</i>.*?'
            r'<div class="p-img">.*?<img src="(.*?)".*?</div>',
            re.DOTALL
        )

        for match in item_pattern.finditer(html):
            product_id = match.group(1)
            name = self._clean_html(match.group(2))
            price = float(match.group(3)) if match.group(3) else 0
            image = match.group(4) if match.group(4) else ""

            products.append({
                "product_id": product_id,
                "name": name,
                "price": price,
                "image_url": image,
                "platform": "JD",
                "source": f"https://item.jd.com/{product_id}.html",
            })

        return products

    def get_product_detail(self, product_id: str) -> Optional[Dict]:
        """
        Get detailed product information from JD.

        Args:
            product_id: JD product SKU ID

        Returns:
            Detailed product dictionary
        """
        url = f"https://item.jd.com/{product_id}.html"

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
        Parse JD product detail page.

        Args:
            html: Raw HTML content
            product_id: Product ID

        Returns:
            Parsed product details
        """
        detail = {"product_id": product_id, "platform": "JD"}

        # Extract brand
        brand_match = re.search(r'<div class="p-parameter".*?品牌：.*?<a.*?>(.*?)</a>', html, re.DOTALL)
        if brand_match:
            detail["brand"] = brand_match.group(1).strip()

        # Extract color
        color_match = re.search(r'<div class="p-parameter".*?颜色：.*?<a.*?>(.*?)</a>', html, re.DOTALL)
        if color_match:
            detail["color"] = color_match.group(1).strip()

        # Extract size
        size_match = re.search(r'<div class="p-parameter".*?尺码：.*?<a.*?>(.*?)</a>', html, re.DOTALL)
        if size_match:
            detail["size"] = size_match.group(1).strip()

        # Extract category breadcrumb
        cat_pattern = re.compile(r'<div class="crumb".*?<a href=".*?">(.*?)</a>', re.DOTALL)
        categories = cat_pattern.findall(html)
        if categories:
            detail["category"] = " > ".join([c.strip() for c in categories if c.strip()])

        # Extract description
        desc_match = re.search(r'<div class="p-detail".*?<div class="p-parameter".*?</div>', html, re.DOTALL)
        if desc_match:
            detail["description"] = self._clean_html(desc_match.group(0))

        # Extract images
        img_pattern = re.compile(r'<li class="lh-item".*?<img src="(.*?)".*?</li>', re.DOTALL)
        images = img_pattern.findall(html)
        detail["images"] = [img.replace("/n5/", "/n1/") for img in images if img.startswith("//")]

        # Extract comments count
        comment_match = re.search(r'<div class="rate-count">.*?<span>(.*?)</span>', html, re.DOTALL)
        if comment_match:
            detail["comment_count"] = self._extract_number(comment_match.group(1))

        # Extract好评率
        good_rate_match = re.search(r'好评率.*?<i>([\d.]+)%</i>', html, re.DOTALL)
        if good_rate_match:
            detail["good_rate"] = float(good_rate_match.group(1))

        return detail

    def _clean_html(self, text: str) -> str:
        """Remove HTML tags and clean text."""
        if not text:
            return ""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Decode HTML entities
        text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&quot;", '"')
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _extract_number(self, text: str) -> int:
        """Extract number from text."""
        match = re.search(r'[\d,]+', text)
        if match:
            return int(match.group().replace(",", ""))
        return 0

    def crawl_with_progress(
        self,
        categories: List[str],
        pages_per_category: int = 3,
        progress_callback=None
    ) -> List[Dict]:
        """
        Crawl products with progress callback.

        Args:
            categories: List of categories to crawl
            pages_per_category: Pages per category
            progress_callback: Callback function(current, total, product)

        Returns:
            List of all crawled products
        """
        all_products = []
        total = len(categories) * pages_per_category
        current = 0

        for category in categories:
            for page in range(1, pages_per_category + 1):
                current += 1
                products = self.search(category, page, 20)
                all_products.extend(products)

                if progress_callback:
                    progress_callback(current, total, len(products))

                self._random_delay()

        return all_products


# JD Category Keywords for clothing
JD_CLOTHING_CATEGORIES = [
    "T恤男", "T恤女", "衬衫男", "衬衫女",
    "牛仔裤男", "牛仔裤女", "连衣裙", "半身裙",
    "男士外套", "女士外套", "羽绒服", "棉服",
    "运动裤", "休闲裤", "西装", "卫衣",
    "毛衣", "针织衫", "polo衫", "背心",
    "男鞋", "女鞋", "运动鞋", "休闲鞋",
    "皮带", "围巾", "帽子", "手套",
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    scraper = JDScraper()

    # Test search
    print("Testing JD scraper...")
    products = scraper.search("T恤男", page=1)
    print(f"Found {len(products)} products")

    if products:
        print("\nFirst product:")
        print(products[0])

        # Test get detail
        detail = scraper.get_product_detail(products[0]["product_id"])
        if detail:
            print("\nProduct detail:")
            for k, v in detail.items():
                print(f"  {k}: {v}")
