"""
Product Search Reranker Service

This service combines dense and sparse search results using Reciprocal Rank Fusion (RRF),
then reranks products using a simulated ML model for semantic relevance.

Your task:
1. Find and fix the bugs in this code
2. Optimize the inefficient patterns
3. Implement the missing caching layer
4. Design a fallback strategy for when the reranker fails

Run with: uvicorn reranker_service:app --reload --port 8000
Test with: python test_reranker.py
"""

import asyncio
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import polars as pl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# =============================================================================
# Configuration
# =============================================================================

BATCH_SIZE = 10  # Fixed batch size for reranking
MAX_WORKERS = 4  # Thread pool workers
RRF_K = 60  # RRF constant

app = FastAPI(title="Product Reranker Service")


# =============================================================================
# Data Models
# =============================================================================


class Product(BaseModel):
    id: str
    name: str
    description: str
    price: float
    category: str


class RerankerRequest(BaseModel):
    query: str
    query_id: str
    dense_results: dict[str, float]  # product_id -> dense score
    sparse_results: dict[str, float]  # product_id -> sparse score
    products: list[Product]


class RerankerResponse(BaseModel):
    query_id: str
    ranked_products: list[dict]
    latency_ms: float
    rrf_scores: dict[str, float]
    reranker_scores: dict[str, float]


# =============================================================================
# Core Reranking Logic
# =============================================================================


def calculate_rrf(
    dense_scores: dict[str, float],
    sparse_scores: dict[str, float],
    k: int = RRF_K,
) -> dict[str, float]:
    """
    Calculate Reciprocal Rank Fusion (RRF) scores from dense and sparse results.
    
    RRF combines rankings from multiple retrieval systems by:
    RRF(d) = sum(1 / (k + rank_i(d))) for each system i
    
    Args:
        dense_scores: Dictionary mapping product IDs to dense retrieval scores
        sparse_scores: Dictionary mapping product IDs to sparse retrieval scores  
        k: Smoothing constant (typically 60)
    
    Returns:
        Dictionary mapping product IDs to combined RRF scores
    """
    # Sort scores (descending order - higher score = better rank)
    sorted_dense = sorted(dense_scores.items(), key=lambda x: x[1], reverse=True)
    sorted_sparse = sorted(sparse_scores.items(), key=lambda x: x[1], reverse=True)

    # Create rank maps (rank starts at 1)
    rank_map_dense = {id: rank + 1 for rank, (id, _) in enumerate(sorted_dense)}
    rank_map_sparse = {id: rank + 1 for rank, (id, _) in enumerate(sorted_sparse)}

    # Calculate RRF for each product - only include products in BOTH lists
    rrf_scores = {}
    all_ids = set(rank_map_dense.keys()).intersection(set(rank_map_sparse.keys()))
    
    for product_id in all_ids:
        rank_dense = rank_map_dense.get(product_id, float("inf"))
        rank_sparse = rank_map_sparse.get(product_id, float("inf"))
        rrf_score = (1 / (k + rank_dense)) + (1 / (k + rank_sparse))
        rrf_scores[product_id] = rrf_score

    # Sort by RRF score descending
    sorted_rrf = dict(sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True))
    return sorted_rrf


def simulate_reranker_api(query: str, product_text: str) -> float:
    """
    Simulates calling an external ML reranker API (like Google Discovery Engine).
    In production, this would be a cross-encoder model that scores query-product relevance.
    
    Args:
        query: The search query
        product_text: Concatenated product name + description
    
    Returns:
        Relevance score between 0 and 1
    """
    # Simulate API latency (50-150ms per call)
    time.sleep(random.uniform(0.05, 0.15))
    
    # Simulate relevance scoring based on word overlap
    query_words = set(query.lower().split())
    product_words = set(product_text.lower().split())
    overlap = len(query_words & product_words)
    
    # Add some randomness to simulate ML model behavior
    base_score = min(overlap / max(len(query_words), 1), 1.0)
    noise = random.uniform(-0.1, 0.1)
    return max(0, min(1, base_score + noise))


class ProductReranker:
    """
    Handles product reranking using batch processing and parallel execution.
    """
    
    def __init__(self, batch_size: int = BATCH_SIZE, max_workers: int = MAX_WORKERS):
        self.batch_size = batch_size
        self.max_workers = max_workers
        # TODO: Add embedding cache here
    
    def _chunk_products(self, products: list[Product]) -> list[list[Product]]:
        """Split products into batches for parallel processing."""
        chunks = []
        for i in range(0, len(products), self.batch_size):
            chunks.append(products[i:i + self.batch_size])
        return chunks
    
    def _process_batch(self, query: str, batch: list[Product]) -> dict[str, float]:
        """
        Process a single batch of products through the reranker.
        
        Args:
            query: Search query
            batch: List of products to rerank
        
        Returns:
            Dictionary mapping product IDs to reranker scores
        """
        batch_results = {}
        for product in batch:
            product_text = f"{product.name} {product.description}"
            score = simulate_reranker_api(query, product_text)
            batch_results[product.id] = score
        return batch_results
    
    def rerank_products(
        self,
        query: str,
        products: list[Product],
        rrf_scores: dict[str, float],
    ) -> pl.DataFrame:
        """
        Rerank products using parallel batch processing.
        
        Args:
            query: Search query
            products: List of products to rerank
            rrf_scores: Pre-computed RRF scores
        
        Returns:
            Polars DataFrame with products sorted by final score
        """
        if not products:
            return pl.DataFrame()
        
        # Create initial DataFrame
        df = pl.DataFrame([
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "price": p.price,
                "category": p.category,
            }
            for p in products
        ])
        
        # Add RRF scores by creating a lookup series and joining
        rrf_scores_copy = dict(rrf_scores)  # Copy for thread safety
        rrf_df = pl.DataFrame({
            "id": list(rrf_scores_copy.keys()),
            "rrf_score": [float(v) for v in rrf_scores_copy.values()]
        })
        df = df.join(rrf_df, on="id", how="left").with_columns(
            pl.col("rrf_score").fill_null(0.0)
        )
        
        # Chunk products for parallel processing
        chunks = self._chunk_products(products)
        
        # Process chunks in parallel
        reranker_scores = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_batch, query, chunk): chunk 
                for chunk in chunks
            }
            
            for future in as_completed(futures):
                try:
                    chunk_result = future.result()
                    reranker_scores.update(chunk_result)
                except Exception as e:
                    print(f"Batch processing failed: {e}")
        
        # Add reranker scores by creating a lookup series and joining
        reranker_scores_copy = dict(reranker_scores)  # Copy for thread safety
        reranker_df = pl.DataFrame({
            "id": list(reranker_scores_copy.keys()),
            "reranker_score": [float(v) for v in reranker_scores_copy.values()]
        })
        df = df.join(reranker_df, on="id", how="left").with_columns(
            pl.col("reranker_score").fill_null(0.0)
        )
        
        # Calculate final score (weighted combination)
        # Copy DataFrame to add new column
        df_with_scores = df.clone()
        df_with_scores = df_with_scores.with_columns(
            (pl.col("rrf_score") * 0.3 + pl.col("reranker_score") * 0.7).alias("final_score")
        )
        
        # Create another copy for normalization
        df_normalized = df_with_scores.clone()
        
        # Normalize scores to 0-1 range
        max_score = df_normalized.select(pl.col("final_score").max()).item()
        min_score = df_normalized.select(pl.col("final_score").min()).item()
        
        if max_score > min_score:
            df_final = df_normalized.with_columns(
                ((pl.col("final_score") - min_score) / (max_score - min_score)).alias("normalized_score")
            )
        else:
            df_final = df_normalized.with_columns(
                pl.lit(1.0).alias("normalized_score")
            )
        
        # Sort by final score
        df_sorted = df_final.sort("final_score", descending=True)
        
        return df_sorted, reranker_scores


# =============================================================================
# API Endpoints
# =============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/rerank", response_model=RerankerResponse)
async def rerank_products(request: RerankerRequest):
    """
    Rerank products based on query relevance.
    
    Combines dense and sparse retrieval scores using RRF, then applies
    ML-based semantic reranking for final ordering.
    """
    start_time = time.time()
    
    try:
        # Step 1: Calculate RRF scores
        rrf_scores = calculate_rrf(
            request.dense_results,
            request.sparse_results,
        )
        
        # Step 2: Rerank with ML model
        reranker = ProductReranker()
        df_ranked, reranker_scores = reranker.rerank_products(
            request.query,
            request.products,
            rrf_scores,
        )
        
        # Step 3: Format response
        ranked_products = df_ranked.to_dicts() if not df_ranked.is_empty() else []
        
        latency_ms = (time.time() - start_time) * 1000
        
        return RerankerResponse(
            query_id=request.query_id,
            ranked_products=ranked_products,
            latency_ms=round(latency_ms, 2),
            rrf_scores=rrf_scores,
            reranker_scores=reranker_scores,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RRFRequest(BaseModel):
    dense_results: dict[str, float]
    sparse_results: dict[str, float]


@app.post("/rrf-only")
async def calculate_rrf_only(request: RRFRequest):
    """
    Calculate RRF scores without reranking.
    Useful for testing the RRF implementation.
    """
    try:
        rrf_scores = calculate_rrf(request.dense_results, request.sparse_results)
        return {"rrf_scores": rrf_scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Embedding Cache (TODO: Implement)
# =============================================================================


class EmbeddingCache:
    """
    TODO: Implement an in-memory cache for embeddings with TTL support.
    
    Requirements:
    - Cache embeddings by query string (case-insensitive)
    - Support TTL (time-to-live) for cache entries
    - Thread-safe operations
    - Maximum cache size with LRU eviction
    
    Example usage:
        cache = EmbeddingCache(max_size=1000, ttl_seconds=300)
        cache.set("laptop gaming", {"dense": [...], "sparse": [...]})
        result = cache.get("laptop gaming")  # Returns cached value or None
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        # TODO: Implement cache storage
        pass
    
    def get(self, query: str) -> Optional[dict]:
        """Get cached embedding for query."""
        # TODO: Implement
        return None
    
    def set(self, query: str, embedding: dict) -> None:
        """Cache embedding for query."""
        # TODO: Implement
        pass
    
    def clear(self) -> None:
        """Clear all cached entries."""
        # TODO: Implement
        pass


# =============================================================================
# Fallback Strategy (TODO: Design)
# =============================================================================

"""
TODO: Design and implement a fallback strategy for when the reranker API fails.

Consider:
1. What should happen if the simulated reranker API times out?
2. What if it returns errors for some products but not others?
3. How do you ensure the user still gets useful results?
4. How do you log/monitor these failures?

Implement your fallback strategy in the ProductReranker class.
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
