"""Data cleaning and standardization for clothing data."""
import re
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class DataWasher:
    """Clean and standardize clothing product data."""

    # Size standardization mapping
    SIZE_MAPPING = {
        "XS": ["xs", "XS", "特小", "小号", "155/80A", "155/80S"],
        "S": ["s", "S", "小", "中号", "155/80A", "160/84A", "165/88A"],
        "M": ["m", "M", "中", "160/84A", "165/88A", "165/84A"],
        "L": ["l", "L", "大", "170/92A", "170/88A", "175/96A"],
        "XL": ["xl", "XL", "加大", "175/96A", "175/100A", "180/104A"],
        "XXL": ["xxl", "XXL", "特大", "180/104A", "185/108A"],
        "XXXL": ["xxxl", "XXXL", "超大", "185/108A", "190/112A"],
    }

    # Material standardization mapping
    MATERIAL_MAPPING = {
        "棉": ["棉", "纯棉", "100%棉", "全棉", "cotton"],
        "麻": ["麻", "亚麻", "苎麻", "linen", "hemp"],
        "丝绸": ["丝绸", "真丝", "桑蚕丝", "silk"],
        "羊毛": ["羊毛", "羊绒", "毛", "wool"],
        "涤纶": ["涤纶", "聚酯纤维", "polyester", "聚酯"],
        "牛仔布": ["牛仔布", "牛仔裤面料", "denim"],
        "皮革": ["皮革", "真皮", "皮", "leather"],
        "蕾丝": ["蕾丝", "花边", "lace"],
    }

    # Color standardization mapping
    COLOR_MAPPING = {
        "黑色": ["黑", "黑色", "纯黑", "炭黑", "black"],
        "白色": ["白", "白色", "纯白", "米白", "white"],
        "灰色": ["灰", "灰色", "银灰", "深灰", "浅灰", "gray", "grey"],
        "蓝色": ["蓝", "蓝色", "天蓝", "湖蓝", "藏蓝", "宝蓝", "深蓝", "浅蓝", "blue", "navy"],
        "红色": ["红", "红色", "酒红", "枣红", "砖红", "粉红", "玫红", "red", "pink"],
        "绿色": ["绿", "绿色", "军绿", "墨绿", "浅绿", "草绿", "green"],
        "黄色": ["黄", "黄色", "鹅黄", "姜黄", "土黄", "金黄", "yellow", "gold"],
        "粉色": ["粉", "粉色", "藕粉", "肉粉", "pink"],
        "紫色": ["紫", "紫色", "紫罗兰", "香芋紫", "purple", "violet"],
        "橙色": ["橙", "橙色", "橘色", "桔色", "orange"],
        "棕色": ["棕", "棕色", "驼色", "卡其", "褐色", "brown", "tan", "khaki"],
        "米色": ["米", "米色", "杏色", "beige", "cream"],
        "白色": ["白", "白色", "象牙白", "white", "ivory"],
    }

    # Category mapping
    CATEGORY_MAPPING = {
        "top": ["T恤", "T恤衫", "短袖", "长袖", "Polo", " polo", "衬衫", "衬衣"],
        "bottom": ["裤", "裙子", "裙", "牛仔裤", "休闲裤", "运动裤", "短裙", "长裙"],
        "dress": ["连衣裙", "裙", "连衣裤"],
        "outerwear": ["外套", "夹克", "大衣", "风衣", "羽绒服", "棉服", "毛衣", "针织衫"],
        "shoes": ["鞋", "靴", "凉鞋", "拖鞋", "运动鞋", "皮鞋", "帆布鞋"],
        "accessories": ["包", "帽", "围巾", "皮带", "手套", "袜子", "项链", "耳环", "手表"],
    }

    # Style keywords
    STYLE_KEYWORDS = {
        "minimalist": ["简约", "极简", "基本", "纯色", "基础款"],
        "casual": ["休闲", "日常", "通勤", "舒适"],
        "bohemian": ["波西米亚", "度假", "民族", "异域"],
        "vintage": ["复古", "vintage", "怀旧", "古典"],
        "streetwear": ["街头", "潮流", "嘻哈", "工装"],
        "business": ["商务", "正装", "职业", "西装"],
        "sporty": ["运动", "健身", "功能性", "体育"],
        "romantic": ["浪漫", "优雅", "淑女", "温柔"],
    }

    # Season keywords
    SEASON_KEYWORDS = {
        "spring": ["春", "春季", "早春"],
        "summer": ["夏", "夏季", "春夏", "清凉", "薄款"],
        "fall": ["秋", "秋季", "秋冬", "深秋"],
        "winter": ["冬", "冬季", "保暖", "加绒", "加厚"],
        "all_season": ["四季", "通用", "百搭", "全年"],
    }

    def __init__(self):
        """Initialize data washer with compiled patterns."""
        self.text_clean_pattern = re.compile(
            r'[^一-龥a-zA-Z0-9，。！？、；："\'（）《》【】\s]'
        )
        self.price_pattern = re.compile(r'¥?([\d.]+)')

        # Build reverse lookup tables
        self.size_reverse = {}
        for std, variants in self.SIZE_MAPPING.items():
            for v in variants:
                self.size_reverse[v.lower()] = std

        self.material_reverse = {}
        for std, variants in self.MATERIAL_MAPPING.items():
            for v in variants:
                self.material_reverse[v] = std

        self.color_reverse = {}
        for std, variants in self.COLOR_MAPPING.items():
            for v in variants:
                self.color_reverse[v] = std

    def clean_text(self, text: str) -> str:
        """
        Deep clean product text.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Replace HTML entities FIRST, while & and ; are still intact
        # (text_clean_pattern below strips & and ;, which would break the matches)
        replacements = {
            '&nbsp;': ' ',
            '&amp;': '&',
            '&quot;': '"',
            '&#39;': "'",
            '&lt;': '<',
            '&gt;': '>',
            '\\n': ' ',
            '\\t': ' ',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove special characters but keep Chinese, English, numbers, punctuation
        text = self.text_clean_pattern.sub('', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def standardize_size(self, size_str: str) -> Optional[str]:
        """
        Standardize size value.

        Args:
            size_str: Raw size string

        Returns:
            Standardized size (XS/S/M/L/XL/XXL/XXXL) or None
        """
        if not size_str:
            return None

        size_str = str(size_str).lower().strip()

        # Direct lookup
        if size_str in self.size_reverse:
            return self.size_reverse[size_str]

        # Fuzzy match using Levenshtein-like approach
        best_match = None
        best_score = 0
        threshold = 0.75

        for variant, std in self.size_reverse.items():
            score = self._similarity(size_str, variant)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = std

        return best_match

    def standardize_material(self, material_str: str) -> Optional[str]:
        """
        Standardize material value.

        Args:
            material_str: Raw material string

        Returns:
            Standardized material or None
        """
        if not material_str:
            return None

        material_str = self.clean_text(material_str)

        # Check each known material
        for std, keywords in self.MATERIAL_MAPPING.items():
            for keyword in keywords:
                if keyword in material_str:
                    return std

        # Try to extract using pattern
        match = re.search(r'(棉|麻|丝绸|羊毛|涤纶|牛仔|皮革|蕾丝)', material_str)
        if match:
            keyword = match.group(1)
            return self.material_reverse.get(keyword)

        return None

    def standardize_color(self, color_str: str) -> Optional[str]:
        """
        Standardize color value.

        Args:
            color_str: Raw color string

        Returns:
            Standardized color or None
        """
        if not color_str:
            return None

        color_str = self.clean_text(color_str)

        # Check each known color
        for std, keywords in self.COLOR_MAPPING.items():
            for keyword in keywords:
                if keyword in color_str:
                    return std

        return None

    def standardize_category(self, category_str: str) -> Optional[str]:
        """
        Standardize product category.

        Args:
            category_str: Raw category string

        Returns:
            Standardized category (top/bottom/dress/outerwear/shoes/accessories)
        """
        if not category_str:
            return None

        category_str = self.clean_text(category_str)

        for std, keywords in self.CATEGORY_MAPPING.items():
            for keyword in keywords:
                if keyword in category_str:
                    return std

        return None

    def extract_style(self, text: str) -> List[str]:
        """
        Extract style tags from text.

        Args:
            text: Product text (name, description, etc.)

        Returns:
            List of detected styles
        """
        if not text:
            return []

        text = self.clean_text(text)
        styles = []

        for style, keywords in self.STYLE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    styles.append(style)
                    break

        return list(set(styles))  # Remove duplicates

    def extract_season(self, text: str) -> Optional[str]:
        """
        Extract season from text.

        Args:
            text: Product text

        Returns:
            Detected season or None
        """
        if not text:
            return None

        text = self.clean_text(text)

        for season, keywords in self.SEASON_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    return season

        return None

    def extract_price(self, price_str: str) -> Optional[float]:
        """
        Extract numeric price from string.

        Args:
            price_str: Raw price string

        Returns:
            Numeric price or None
        """
        if not price_str:
            return None

        # Try to find price pattern
        match = self.price_pattern.search(str(price_str))
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass

        return None

    def clean_product(self, product: Dict) -> Dict:
        """
        Clean and standardize a single product.

        Args:
            product: Raw product dictionary

        Returns:
            Cleaned product dictionary
        """
        cleaned = {
            "product_id": str(product.get("product_id", "")),
            "name": self.clean_text(product.get("name", "")),
            "platform": product.get("platform", "UNKNOWN"),
            "source": product.get("source", ""),
        }

        # Standardize price
        price = self.extract_price(product.get("price", 0))
        if price:
            # Clip to reasonable range [1, 10000]
            cleaned["price"] = max(1.0, min(10000.0, price))
        else:
            cleaned["price"] = 0

        # Standardize other fields
        cleaned["brand"] = self.clean_text(product.get("brand", ""))
        cleaned["category"] = self.standardize_category(product.get("category", ""))
        cleaned["color"] = self.standardize_color(product.get("color", ""))
        cleaned["size"] = self.standardize_size(product.get("size", ""))
        cleaned["material"] = self.standardize_material(product.get("material", ""))

        # Extract styles
        name = cleaned["name"]
        desc = self.clean_text(product.get("description", ""))
        combined_text = f"{name} {desc}"
        cleaned["styles"] = self.extract_style(combined_text)
        cleaned["style"] = cleaned["styles"][0] if cleaned["styles"] else "casual"

        # Extract season
        cleaned["season"] = self.extract_season(combined_text) or "all_season"

        # Store original values for reference
        cleaned["_original_name"] = product.get("name", "")
        cleaned["_original_category"] = product.get("category", "")

        return cleaned

    def clean_batch(self, products: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Clean a batch of products with statistics.

        Args:
            products: List of raw products

        Returns:
            Tuple of (cleaned products, statistics dict)
        """
        cleaned_products = []
        stats = {
            "total": len(products),
            "cleaned": 0,
            "with_category": 0,
            "with_color": 0,
            "with_size": 0,
            "with_material": 0,
            "with_style": 0,
            "with_season": 0,
            "duplicates_removed": 0,
        }

        seen_ids = set()
        duplicates = 0

        for product in products:
            # Check for duplicate
            pid = str(product.get("product_id", ""))
            if pid in seen_ids:
                duplicates += 1
                continue
            seen_ids.add(pid)

            cleaned = self.clean_product(product)
            cleaned_products.append(cleaned)

            # Track statistics
            if cleaned["category"]:
                stats["with_category"] += 1
            if cleaned["color"]:
                stats["with_color"] += 1
            if cleaned["size"]:
                stats["with_size"] += 1
            if cleaned["material"]:
                stats["with_material"] += 1
            if cleaned["styles"]:
                stats["with_style"] += 1
            if cleaned["season"]:
                stats["with_season"] += 1

        stats["duplicates_removed"] = duplicates
        stats["cleaned"] = len(cleaned_products)

        logger.info(
            f"Cleaned {stats['cleaned']}/{stats['total']} products, "
            f"removed {duplicates} duplicates. "
            f"Category: {stats['with_category']}, "
            f"Style: {stats['with_style']}, "
            f"Season: {stats['with_season']}"
        )

        return cleaned_products, stats

    def _similarity(self, s1: str, s2: str) -> float:
        """
        Calculate simple string similarity.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Similarity score [0, 1]
        """
        if not s1 or not s2:
            return 0

        # Simple character-level Jaccard similarity
        set1 = set(s1)
        set2 = set(s2)

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 0

        return intersection / union


# CLI for testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    washer = DataWasher()

    # Test cleaning
    test_products = [
        {
            "product_id": "12345",
            "name": "夏季新款简约纯棉T恤男 白色 黑色 多颜色可选",
            "price": "¥129.00",
            "brand": "Nike",
            "category": "T恤",
            "platform": "JD",
        },
        {
            "product_id": "12346",
            "name": "2024新款波西米亚风格连衣裙 女 碎花 雪纺",
            "price": 299,
            "brand": "ZARA",
            "category": "裙子",
            "platform": "TMALL",
        },
    ]

    print("Testing data cleaning...")
    cleaned, stats = washer.clean_batch(test_products)

    print(f"\nStatistics: {stats}")
    print("\nCleaned products:")
    for p in cleaned:
        print(f"  {p['product_id']}: {p['name']}")
        print(f"    Category: {p['category']}, Color: {p['color']}, "
              f"Style: {p['style']}, Season: {p['season']}")
        print(f"    Material: {p['material']}, Size: {p['size']}, Price: {p['price']}")
