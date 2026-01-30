# ML Engineering Interview: Product Search Reranker

Welcome to the ML Engineering coding interview! This exercise tests your ability to debug, optimize, and extend a real-world product search reranking service.

**Time:** 60 minutes  
**Tools:** You may use any LLM tools (Copilot, Claude, ChatGPT, etc.) to help you

---

## Background

You're working on a product search system that combines two retrieval methods:
1. **Dense retrieval** - Using vector embeddings (semantic similarity)
2. **Sparse retrieval** - Using keyword matching (BM25/TF-IDF)

The results from both methods are combined using **Reciprocal Rank Fusion (RRF)**, then reranked using an ML model for final ordering.

This service has been deployed to production but users are reporting issues. Your task is to identify and fix the bugs, then optimize the code.

---

## Setup

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Service

```bash
# Start the server
uvicorn reranker_service:app --reload --port 8000

# In another terminal, run tests
python test_reranker.py
```

The service will be available at `http://localhost:8000`

### API Documentation

Once running, visit `http://localhost:8000/docs` for interactive API docs.

---

## Your Tasks

### Part 1: Debugging

The test suite (`test_reranker.py`) will fail several tests. Your task is to:

1. **Find Bug #1:** The RRF calculation is dropping products unexpectedly
   - Some products that should appear in results are missing entirely

2. **Find Bug #2:** The RRF returns unfair scores for certain products
   - Products appearing in only one retrieval system are being penalized too harshly

For each bug:
- Identify the root cause
- Implement a fix
- Explain why the bug occurred

### Part 2: Code Review

Review the `rerank_products()` method, specifically the ThreadPoolExecutor usage:
- Is the code thread-safe?
- What could go wrong in other Python implementations?
- Would you approve this in a code review? Why or why not?
- If you see issues, propose a fix

### Part 3: Architecture

1. **Implement Embedding Cache**
   - Complete the `EmbeddingCache` class in `reranker_service.py`
   - Requirements: TTL support, thread-safe, LRU eviction
   - Integrate it into the reranking flow

2. **Design Fallback Strategy**
   - What should happen when `simulate_reranker_api()` fails?
   - Implement graceful degradation with retry logic

### Part 4: Efficiency

1. **Optimize Batch Processing**
   - The current batch size is fixed at 10
   - Analyze and implement a better batching strategy

2. **Reduce Memory Overhead**
   - Find and fix unnecessary DataFrame copies
   - Optimize the DataFrame operations

### Part 5: Code Quality 

- Clean, readable code
- Appropriate comments
- Proper error handling

---

## Hints

### RRF Formula
```
RRF(d) = Σ 1/(k + rank_i(d))
```
Where:
- `k` is a constant (typically 60)
- `rank_i(d)` is the rank of document `d` in the i-th ranking (1-indexed)

### Key Files
- `reranker_service.py` - Main service code (edit this)
- `test_reranker.py` - Test suite (run this to check your work)

### Running Individual Tests
```bash
# Run all tests
python test_reranker.py

# Or use pytest for detailed output
pytest test_reranker.py -v

# Run specific test
pytest test_reranker.py -v -k "test_rrf_edge_case"
```


## Submission

When you're done:
1. Ensure all tests pass: `python test_reranker.py`
2. Be prepared to explain your changes and design decisions

Good luck!

---

## Quick Reference

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/rerank` | POST | Full reranking pipeline |
| `/rrf-only` | POST | Calculate RRF scores only |

### Sample Request

```bash
curl -X POST "http://localhost:8000/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "gaming laptop",
    "query_id": "test-001",
    "dense_results": {"SKU001": 0.95, "SKU002": 0.80},
    "sparse_results": {"SKU001": 0.90, "SKU002": 0.85},
    "products": [
      {"id": "SKU001", "name": "Gaming Laptop", "description": "High performance", "price": 1999.99, "category": "Laptops"},
      {"id": "SKU002", "name": "Business Laptop", "description": "Professional use", "price": 1299.99, "category": "Laptops"}
    ]
  }'
```

### Expected Response

```json
{
  "query_id": "test-001",
  "ranked_products": [...],
  "latency_ms": 250.5,
  "rrf_scores": {"SKU001": 0.0328, "SKU002": 0.0323},
  "reranker_scores": {"SKU001": 0.85, "SKU002": 0.72}
}
```
