"""Knowledge-graph visualization endpoints.

Provides aggregate stats and a sampled subgraph for the frontend
force-directed visualization.
"""
from fastapi import APIRouter, Query, HTTPException

from api.database import get_neo4j_driver
from kg.graph.query import ClothingGraphQuery


router = APIRouter(prefix="/api/kg", tags=["knowledge-graph"])


def _query() -> ClothingGraphQuery:
    driver = get_neo4j_driver()
    if driver is None:
        raise HTTPException(status_code=503, detail="Neo4j driver not available")
    return ClothingGraphQuery(driver)


@router.get("/stats")
def get_stats():
    """Aggregate node / relationship counts (lightweight)."""
    return _query().get_graph_stats()


@router.get("/graph")
def get_graph(limit: int = Query(120, ge=10, le=500)):
    """Sampled subgraph for force-directed visualization.

    ``limit`` caps the number of *nodes* returned; edges are restricted
    to those whose endpoints both fall inside the sample.
    """
    return _query().get_subgraph(limit=limit)
