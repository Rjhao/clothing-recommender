"""Clothing service for managing clothing items."""
from typing import List, Optional
from neo4j import Driver

from kg.graph.builder import ClothingGraphBuilder
from kg.graph.query import ClothingGraphQuery
from kg.image_url import resolve_image as _resolve_image


def _clothing_id(item: dict) -> int:
    """Read clothing id tolerating either 'id' (int) or 'product_id' (str) schemas."""
    raw = item.get("id")
    if raw is None:
        raw = item.get("product_id", 0)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


# Image URL resolution is shared with the recommendation engine — see
# kg.image_url.resolve_image (imported above as _resolve_image).


class ClothingService:
    """Service for managing clothing items in the knowledge graph."""

    def __init__(self, driver: Driver):
        """Initialize with Neo4j driver."""
        self.driver = driver
        self.builder = ClothingGraphBuilder(driver)
        self.query = ClothingGraphQuery(driver)

    def create_clothing(self, clothing_data: dict) -> dict:
        """Create a new clothing item."""
        return self.builder.create_clothing(clothing_data)

    def get_clothing(self, clothing_id: int) -> Optional[dict]:
        """Get a clothing item by ID."""
        query = """
        MATCH (c:Clothing {id: $clothing_id})
        RETURN c
        """
        with self.driver.session() as session:
            result = session.run(query, clothing_id=clothing_id)
            record = result.single()
            if record:
                data = dict(record["c"])
                data["image_url"] = _resolve_image(data)
                return data
            return None

    def list_clothing(
        self,
        category: Optional[str] = None,
        style: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[dict]:
        """List clothing items with optional filters."""
        # Empty string (sent by frontend for "no filter") must be treated as None,
        # otherwise the Cypher WHERE clause matches nothing and the list is empty.
        category = category or None
        style = style or None
        results = self.query.find_by_category_style(
            category=category,
            style=style,
            offset=offset,
            limit=limit
        )
        # Transform to match API response format
        transformed = []
        for item in results:
            transformed.append({
                "id": _clothing_id(item),
                "name": item.get("name", ""),
                "category": item.get("category", "top"),
                "brand": item.get("brand"),
                "color": item.get("color"),
                "material": item.get("material"),
                "style": item.get("style"),
                "season": item.get("season"),
                "occasion": item.get("occasion"),
                "price": item.get("price"),
                "image_url": _resolve_image(item),
            })
        return transformed

    def get_similar(self, clothing_id: int, limit: int = 10) -> List[dict]:
        """Get similar clothing items."""
        results = self.query.get_similar_clothing(clothing_id, limit=limit)
        transformed = []
        for item in results:
            similar = item.get("similar", {})
            transformed.append({
                "id": _clothing_id(similar),
                "name": similar.get("name", ""),
                "category": similar.get("category", "top"),
                "brand": similar.get("brand"),
                "color": similar.get("color"),
                "material": similar.get("material"),
                "style": similar.get("style"),
                "season": similar.get("season"),
                "occasion": similar.get("occasion"),
                "price": similar.get("price"),
                "image_url": _resolve_image(similar),
            })
        return transformed

    def get_matching(self, clothing_id: int, limit: int = 10) -> List[dict]:
        """Get matching clothing items."""
        results = self.query.get_matching_outfits(clothing_id, limit=limit)
        transformed = []
        for item in results:
            match = item.get("match", {})
            transformed.append({
                "id": _clothing_id(match),
                "name": match.get("name", ""),
                "category": match.get("category", "top"),
                "brand": match.get("brand"),
                "color": match.get("color"),
                "material": match.get("material"),
                "style": match.get("style"),
                "season": match.get("season"),
                "occasion": match.get("occasion"),
                "price": match.get("price"),
                "image_url": _resolve_image(match),
            })
        return transformed