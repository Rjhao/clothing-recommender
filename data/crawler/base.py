"""Base scraper class for clothing e-commerce platforms."""
import time
import random
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from urllib.parse import quote

import requests
from fake_useragent import UserAgent

logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """Abstract base class for e-commerce scrapers."""

    def __init__(self, delay_range: tuple = (1, 3)):
        """
        Initialize base scraper.

        Args:
            delay_range: Random delay range between requests (min, max) seconds
        """
        self.delay_range = delay_range
        self.ua = UserAgent()
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
        })

    def _random_delay(self):
        """Wait random delay between requests to avoid detection."""
        delay = random.uniform(*self.delay_range)
        time.sleep(delay)

    def _get_headers(self) -> dict:
        """Get random headers for requests."""
        return {
            "User-Agent": self.ua.random,
            "Referer": self.base_url,
        }

    def _get_page(self, url: str, params: Optional[dict] = None) -> Optional[dict]:
        """
        Fetch page content with retry logic.

        Args:
            url: Target URL
            params: Query parameters

        Returns:
            Response JSON or None if failed
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=self._get_headers(),
                    timeout=10
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    self._random_delay()
        return None

    @abstractmethod
    def search(self, keyword: str, page: int = 1, per_page: int = 20) -> List[Dict]:
        """
        Search products by keyword.

        Args:
            keyword: Search keyword
            page: Page number
            per_page: Items per page

        Returns:
            List of product dictionaries
        """
        pass

    @abstractmethod
    def get_product_detail(self, product_id: str) -> Optional[Dict]:
        """
        Get detailed product information.

        Args:
            product_id: Product ID

        Returns:
            Product detail dictionary or None
        """
        pass

    def crawl_category(
        self,
        category: str,
        pages: int = 5,
        per_page: int = 20
    ) -> List[Dict]:
        """
        Crawl products from a category.

        Args:
            category: Category name
            pages: Number of pages to crawl
            per_page: Items per page

        Returns:
            List of crawled products
        """
        all_products = []
        keyword = category  # requests encodes params via params=; do not pre-quote

        for page in range(1, pages + 1):
            logger.info(f"Crawling {category} - page {page}/{pages}")
            products = self.search(keyword, page, per_page)
            all_products.extend(products)
            self._random_delay()

        logger.info(f"Crawled {len(all_products)} products for category '{category}'")
        return all_products

    def crawl_multiple_categories(
        self,
        categories: List[str],
        pages_per_category: int = 3
    ) -> List[Dict]:
        """
        Crawl products from multiple categories.

        Args:
            categories: List of category names
            pages_per_category: Pages to crawl per category

        Returns:
            Combined list of all crawled products
        """
        all_products = []
        for category in categories:
            products = self.crawl_category(category, pages=pages_per_category)
            all_products.extend(products)
            self._random_delay()

        # Deduplicate by product_id
        seen_ids = set()
        unique_products = []
        for product in all_products:
            if product.get("product_id") not in seen_ids:
                seen_ids.add(product.get("product_id"))
                unique_products.append(product)

        logger.info(f"Total unique products: {len(unique_products)}")
        return unique_products
