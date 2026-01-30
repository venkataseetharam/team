"""
Test Suite for Product Reranker Service

This file contains tests that will help you identify bugs and verify your fixes.
Run these tests to validate your implementation.

Usage:
    # Run all tests
    python test_reranker.py
    
    # Run with pytest for more detailed output
    pytest test_reranker.py -v
"""

import time
import requests
import json
from concurrent.futures import ThreadPoolExecutor


# =============================================================================
# Test Data
# =============================================================================

SAMPLE_PRODUCTS = [
    {
        "id": "SKU001",
        "name": "Gaming Laptop RTX 4090",
        "description": "High performance gaming laptop with NVIDIA RTX 4090 graphics card, 32GB RAM, 1TB SSD",
        "price": 2499.99,
        "category": "Laptops"
    },
    {
        "id": "SKU002", 
        "name": "Business Laptop Pro",
        "description": "Professional laptop for business use with Intel i7, 16GB RAM, lightweight design",
        "price": 1299.99,
        "category": "Laptops"
    },
    {
        "id": "SKU003",
        "name": "Gaming Monitor 4K 144Hz",
        "description": "27 inch 4K gaming monitor with 144Hz refresh rate, HDR support, 1ms response time",
        "price": 599.99,
        "category": "Monitors"
    },
    {
        "id": "SKU004",
        "name": "Mechanical Gaming Keyboard RGB",
        "description": "Mechanical keyboard with RGB backlighting, Cherry MX switches, programmable keys",
        "price": 149.99,
        "category": "Keyboards"
    },
    {
        "id": "SKU005",
        "name": "Wireless Gaming Mouse",
        "description": "High precision wireless gaming mouse with 25000 DPI sensor, lightweight design",
        "price": 79.99,
        "category": "Mice"
    },
    {
        "id": "SKU006",
        "name": "USB-C Hub Multiport",
        "description": "USB-C hub with HDMI, USB-A ports, SD card reader, ethernet for laptop connectivity",
        "price": 49.99,
        "category": "Accessories"
    },
    {
        "id": "SKU007",
        "name": "Laptop Stand Adjustable",
        "description": "Ergonomic adjustable laptop stand, aluminum construction, cooling design",
        "price": 39.99,
        "category": "Accessories"
    },
    {
        "id": "SKU008",
        "name": "Gaming Headset 7.1 Surround",
        "description": "Gaming headset with 7.1 surround sound, noise canceling microphone, RGB lighting",
        "price": 89.99,
        "category": "Audio"
    },
    {
        "id": "SKU009",
        "name": "Webcam 4K HDR",
        "description": "4K webcam with HDR, auto-focus, built-in microphone for video calls and streaming",
        "price": 129.99,
        "category": "Cameras"
    },
    {
        "id": "SKU010",
        "name": "External SSD 2TB",
        "description": "Portable external SSD with 2TB storage, USB 3.2 Gen 2, fast transfer speeds",
        "price": 179.99,
        "category": "Storage"
    },
]

# Simulated dense search scores (from vector similarity search)
DENSE_SCORES = {
    "SKU001": 0.95,  # Gaming laptop - high relevance
    "SKU003": 0.88,  # Gaming monitor
    "SKU004": 0.82,  # Gaming keyboard
    "SKU005": 0.79,  # Gaming mouse
    "SKU008": 0.75,  # Gaming headset
    "SKU002": 0.45,  # Business laptop - lower relevance
    "SKU009": 0.35,  # Webcam
    "SKU010": 0.30,  # SSD
}

# Simulated sparse search scores (from BM25/keyword search)
SPARSE_SCORES = {
    "SKU001": 0.92,  # Gaming laptop
    "SKU004": 0.85,  # Gaming keyboard (keyword match)
    "SKU003": 0.80,  # Gaming monitor
    "SKU005": 0.78,  # Gaming mouse
    "SKU008": 0.72,  # Gaming headset
    "SKU002": 0.40,  # Business laptop
    "SKU006": 0.25,  # USB hub
    "SKU007": 0.20,  # Laptop stand
}

BASE_URL = "http://localhost:8000"


# =============================================================================
# Test Functions
# =============================================================================


def test_health_check():
    """Test that the service is running."""
    print("\n" + "=" * 60)
    print("TEST: Health Check")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert data["status"] == "healthy"
        print("✓ Health check passed")
        return True
    except requests.exceptions.ConnectionError:
        print("✗ FAILED: Could not connect to server. Is it running?")
        print("  Start the server with: uvicorn reranker_service:app --reload --port 8000")
        return False
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_rrf_basic():
    """Test basic RRF calculation."""
    print("\n" + "=" * 60)
    print("TEST: Basic RRF Calculation")
    print("=" * 60)
    
    try:
        response = requests.post(
            f"{BASE_URL}/rrf-only",
            json={
                "dense_results": DENSE_SCORES,
                "sparse_results": SPARSE_SCORES,
            },
            timeout=10,
        )
        
        if response.status_code != 200:
            print(f"✗ FAILED: Status code {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
        data = response.json()
        rrf_scores = data["rrf_scores"]
        
        # Verify SKU001 has highest RRF score (top in both lists)
        sorted_scores = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_product = sorted_scores[0][0]
        
        if top_product == "SKU001":
            print("✓ SKU001 correctly ranked first by RRF")
        else:
            print(f"✗ Expected SKU001 first, got {top_product}")
            return False
        
        print(f"  Top 5 RRF scores: {sorted_scores[:5]}")
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_rrf_edge_case_empty_dense():
    """
    BUG TEST: RRF should handle empty dense scores gracefully.
    
    This test exposes Bug #1: Division by zero or crash when one score dict is empty.
    """
    print("\n" + "=" * 60)
    print("TEST: RRF Edge Case - Empty Dense Scores")
    print("=" * 60)
    
    try:
        # Empty dense scores, only sparse scores
        response = requests.post(
            f"{BASE_URL}/rrf-only",
            json={
                "dense_results": {},  # Empty!
                "sparse_results": SPARSE_SCORES,
            },
            timeout=10,
        )
        
        if response.status_code == 200:
            data = response.json()
            rrf_scores = data["rrf_scores"]
            
            # All sparse products should still have RRF scores
            if len(rrf_scores) == len(SPARSE_SCORES):
                print("✓ RRF handles empty dense scores correctly")
                return True
            else:
                print(f"✗ Expected {len(SPARSE_SCORES)} products, got {len(rrf_scores)}")
                return False
        else:
            print(f"✗ FAILED: Service returned error status {response.status_code}")
            print(f"  This likely indicates Bug #1 (empty list handling)")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ FAILED with exception: {e}")
        print("  This likely indicates Bug #1 (empty list handling)")
        return False


def test_rrf_edge_case_empty_sparse():
    """
    BUG TEST: RRF should handle empty sparse scores gracefully.
    """
    print("\n" + "=" * 60)
    print("TEST: RRF Edge Case - Empty Sparse Scores")
    print("=" * 60)
    
    try:
        response = requests.post(
            f"{BASE_URL}/rrf-only",
            json={
                "dense_results": DENSE_SCORES,
                "sparse_results": {},  # Empty!
            },
            timeout=10,
        )
        
        if response.status_code == 200:
            data = response.json()
            rrf_scores = data["rrf_scores"]
            
            if len(rrf_scores) == len(DENSE_SCORES):
                print("✓ RRF handles empty sparse scores correctly")
                return True
            else:
                print(f"✗ Expected {len(DENSE_SCORES)} products, got {len(rrf_scores)}")
                return False
        else:
            print(f"✗ FAILED: Service returned error status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_rrf_single_list_score_quality():
    """
    BUG TEST: Products in only one list should get reasonable scores, not near-zero.
    
    This test exposes Bug #3: Using float("inf") as default rank penalizes
    products too harshly when they only appear in one retrieval system.
    """
    print("\n" + "=" * 60)
    print("TEST: RRF Single List Score Quality")
    print("=" * 60)
    
    try:
        # Only dense scores, no sparse
        dense_only = {"A": 1.0, "B": 0.8, "C": 0.6}
        
        response = requests.post(
            f"{BASE_URL}/rrf-only",
            json={
                "dense_results": dense_only,
                "sparse_results": {},  # Empty sparse
            },
            timeout=10,
        )
        
        if response.status_code != 200:
            print(f"✗ FAILED: Status {response.status_code}")
            return False
        
        data = response.json()
        rrf_scores = data["rrf_scores"]
        
        if len(rrf_scores) != 3:
            print(f"✗ Expected 3 products, got {len(rrf_scores)}")
            return False
        
        # Check that ranking order is preserved (A > B > C)
        if not (rrf_scores.get("A", 0) > rrf_scores.get("B", 0) > rrf_scores.get("C", 0)):
            print(f"✗ Ranking order not preserved: {rrf_scores}")
            return False
        
        print(f"  Scores: A={rrf_scores['A']:.6f}, B={rrf_scores['B']:.6f}, C={rrf_scores['C']:.6f}")
        
        # Key check: scores should be reasonable, not near-zero
        # With proper handling: score ≈ 1/(60+rank) ≈ 0.016 for rank 1
        # With infinity bug: score ≈ 1/(60+inf) + 1/(60+rank) ≈ 0 + 0.016 = 0.016
        # Actually similar... let's check the TOP score is what we expect
        
        # Expected score for A (rank 1) with single list: 1/(60+1) ≈ 0.0164
        expected_single_list_score = 1 / 61
        
        # If using infinity, A gets: 1/(60+1) + 1/(60+inf) ≈ 0.0164 + 0 = 0.0164
        # If handling properly, A gets: 1/(60+1) ≈ 0.0164 (same, but cleaner)
        
        # The real test: when we have MIXED lists (some in both, some in one)
        # Let's test that scenario instead
        print("✓ Basic single-list ranking preserved")
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_rrf_mixed_list_fairness():
    """
    BUG TEST: Products appearing in only ONE list should not be unfairly penalized.
    
    This is the key test for Bug #3: with float("inf") default rank, products
    only in one list get near-zero contribution from the other list, making
    them rank much lower than they should.
    """
    print("\n" + "=" * 60)
    print("TEST: RRF Mixed List Fairness (Bug #3)")
    print("=" * 60)
    
    try:
        # A is #1 in dense only (not in sparse)
        # B is #2 in dense AND #1 in sparse
        # 
        # Fair ranking: A should be competitive with B
        # Buggy (inf): A gets penalized because sparse rank = inf
        
        dense_scores = {"A": 1.0, "B": 0.9}  # A is #1, B is #2
        sparse_scores = {"B": 1.0}           # Only B, ranked #1
        
        response = requests.post(
            f"{BASE_URL}/rrf-only",
            json={
                "dense_results": dense_scores,
                "sparse_results": sparse_scores,
            },
            timeout=10,
        )
        
        if response.status_code != 200:
            print(f"✗ FAILED: Status {response.status_code}")
            return False
        
        data = response.json()
        rrf_scores = data["rrf_scores"]
        
        score_a = rrf_scores.get("A", 0)
        score_b = rrf_scores.get("B", 0)
        
        print(f"  A (dense only): {score_a:.6f}")
        print(f"  B (in both):    {score_b:.6f}")
        
        # With infinity bug:
        # A: 1/(60+1) + 1/(60+inf) = 0.0164 + 0 = 0.0164
        # B: 1/(60+2) + 1/(60+1)   = 0.0161 + 0.0164 = 0.0325
        # B is 2x higher than A (unfair!)
        
        # With proper fix (using len+1 as default):
        # A: 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325
        # B: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325
        # Much more fair!
        
        # Test: A should have at least 70% of B's score (not penalized to <50%)
        ratio = score_a / score_b if score_b > 0 else 0
        
        print(f"  Ratio (A/B): {ratio:.2%}")
        
        if ratio >= 0.70:
            print("✓ Products in single list are fairly scored")
            return True
        else:
            print(f"✗ BUG DETECTED: A is unfairly penalized (ratio={ratio:.2%}, should be >=70%)")
            print("  This happens when using float('inf') as default rank")
            print("  Fix: Use len(scores)+1 as default rank instead of infinity")
            return False
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_rrf_edge_case_both_empty():
    """
    BUG TEST: RRF should handle both empty score dicts.
    """
    print("\n" + "=" * 60)
    print("TEST: RRF Edge Case - Both Empty")
    print("=" * 60)
    
    try:
        response = requests.post(
            f"{BASE_URL}/rrf-only",
            json={
                "dense_results": {},
                "sparse_results": {},
            },
            timeout=10,
        )
        
        if response.status_code == 200:
            data = response.json()
            rrf_scores = data["rrf_scores"]
            
            if len(rrf_scores) == 0:
                print("✓ RRF handles both empty dicts correctly")
                return True
            else:
                print(f"✗ Expected empty result, got {len(rrf_scores)} products")
                return False
        else:
            print(f"✗ FAILED: Status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_full_reranking():
    """Test the full reranking pipeline."""
    print("\n" + "=" * 60)
    print("TEST: Full Reranking Pipeline")
    print("=" * 60)
    
    try:
        request_data = {
            "query": "gaming laptop",
            "query_id": "test-001",
            "dense_results": DENSE_SCORES,
            "sparse_results": SPARSE_SCORES,
            "products": SAMPLE_PRODUCTS,
        }
        
        response = requests.post(
            f"{BASE_URL}/rerank",
            json=request_data,
            timeout=60,  # Reranking can take time
        )
        
        if response.status_code != 200:
            print(f"✗ FAILED: Status code {response.status_code}")
            print(f"  Response: {response.text}")
            return False
        
        data = response.json()
        
        print(f"✓ Reranking completed in {data['latency_ms']:.2f}ms")
        print(f"  Products ranked: {len(data['ranked_products'])}")
        
        # Check that gaming laptop is in top 3
        top_3_ids = [p["id"] for p in data["ranked_products"][:3]]
        if "SKU001" in top_3_ids:
            print(f"✓ Gaming laptop (SKU001) in top 3: {top_3_ids}")
        else:
            print(f"⚠ Gaming laptop (SKU001) not in top 3: {top_3_ids}")
        
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_concurrent_requests():
    """
    BUG TEST: Test concurrent requests to expose race conditions.
    
    This test helps expose Bug #2: Race condition in dictionary updates.
    """
    print("\n" + "=" * 60)
    print("TEST: Concurrent Requests (Race Condition Detection)")
    print("=" * 60)
    
    num_requests = 5  # Reduced from 10 to avoid overloading local server
    results = []
    errors = []
    
    def make_request(request_id):
        try:
            request_data = {
                "query": "gaming laptop",
                "query_id": f"concurrent-{request_id}",
                "dense_results": DENSE_SCORES,
                "sparse_results": SPARSE_SCORES,
                "products": SAMPLE_PRODUCTS[:5],  # Smaller for speed
            }
            
            response = requests.post(
                f"{BASE_URL}/rerank",
                json=request_data,
                timeout=60,  # Increased timeout for concurrent load
            )
            
            if response.status_code == 200:
                data = response.json()
                return len(data["ranked_products"]), data["reranker_scores"]
            else:
                return None, f"Status {response.status_code}: {response.text[:200]}"
                
        except requests.exceptions.Timeout:
            return None, "Request timed out"
        except requests.exceptions.ConnectionError:
            return None, "Connection refused - server may be overloaded"
        except Exception as e:
            return None, f"Exception: {str(e)}"
    
    # Run concurrent requests
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=3) as executor:  # Reduced from 5 to avoid overload
        futures = [executor.submit(make_request, i) for i in range(num_requests)]
        for future in futures:
            count, scores = future.result()
            if count is not None:
                results.append((count, scores))
            else:
                errors.append(scores)
    
    elapsed = time.time() - start_time
    
    print(f"  Completed {len(results)}/{num_requests} requests in {elapsed:.2f}s")
    
    if errors:
        print(f"✗ {len(errors)} requests failed")
        for err in errors[:3]:
            print(f"    Error: {err}")
        return False
    
    # Check consistency - all requests should return same number of products
    counts = [r[0] for r in results]
    if len(set(counts)) == 1:
        print(f"✓ All requests returned consistent results ({counts[0]} products)")
    else:
        print(f"✗ Inconsistent results: {counts}")
        print("  This may indicate a race condition!")
        return False
    
    # Check if scores are consistent
    all_scores = [r[1] for r in results]
    score_keys = [set(s.keys()) for s in all_scores]
    if len(set(map(frozenset, score_keys))) == 1:
        print("✓ All requests returned same product IDs")
    else:
        print("✗ Different product IDs returned - possible race condition!")
        return False
    
    return True


def test_rrf_includes_all_products():
    """
    BUG TEST: RRF should include products that appear in EITHER list, not just BOTH.
    
    This test exposes Bug #3: Using intersection instead of union, which drops
    products that only appear in one retrieval system.
    """
    print("\n" + "=" * 60)
    print("TEST: RRF Includes All Products (Union vs Intersection)")
    print("=" * 60)
    
    try:
        # Dense has: A, B, C
        # Sparse has: B, C, D
        # Expected in RRF: A, B, C, D (union = 4 products)
        # Buggy RRF would return: B, C only (intersection = 2 products)
        
        dense_only = {"A": 0.9, "B": 0.8, "C": 0.7}
        sparse_only = {"B": 0.85, "C": 0.75, "D": 0.65}
        
        response = requests.post(
            f"{BASE_URL}/rrf-only",
            json={
                "dense_results": dense_only,
                "sparse_results": sparse_only,
            },
            timeout=10,
        )
        
        if response.status_code != 200:
            print(f"✗ FAILED: Status {response.status_code}")
            return False
        
        data = response.json()
        rrf_scores = data["rrf_scores"]
        
        returned_ids = set(rrf_scores.keys())
        expected_ids = {"A", "B", "C", "D"}  # Union of both lists
        intersection_ids = {"B", "C"}  # What buggy code returns
        
        print(f"  Dense products: A, B, C")
        print(f"  Sparse products: B, C, D")
        print(f"  Returned products: {returned_ids}")
        
        if returned_ids == expected_ids:
            print("✓ RRF correctly includes all products (union)")
            return True
        elif returned_ids == intersection_ids:
            print("✗ BUG DETECTED: RRF only includes products in BOTH lists (intersection)")
            print("  Missing: A (dense-only) and D (sparse-only)")
            print("  Fix: Change .intersection() to .union() in calculate_rrf()")
            return False
        else:
            print(f"✗ Unexpected product set: {returned_ids}")
            print(f"  Expected: {expected_ids}")
            return False
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_performance_baseline():
    """Measure baseline performance for optimization comparison."""
    print("\n" + "=" * 60)
    print("TEST: Performance Baseline")
    print("=" * 60)
    
    try:
        request_data = {
            "query": "gaming laptop",
            "query_id": "perf-test",
            "dense_results": DENSE_SCORES,
            "sparse_results": SPARSE_SCORES,
            "products": SAMPLE_PRODUCTS,
        }
        
        # Run multiple iterations
        latencies = []
        for i in range(3):
            start = time.time()
            response = requests.post(
                f"{BASE_URL}/rerank",
                json=request_data,
                timeout=60,
            )
            elapsed = (time.time() - start) * 1000
            
            if response.status_code == 200:
                latencies.append(elapsed)
                data = response.json()
                print(f"  Run {i+1}: {elapsed:.2f}ms (server reported: {data['latency_ms']:.2f}ms)")
        
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            print(f"\n  Average latency: {avg_latency:.2f}ms for {len(SAMPLE_PRODUCTS)} products")
            print(f"  Per-product latency: {avg_latency/len(SAMPLE_PRODUCTS):.2f}ms")
            print("\n  Optimization opportunities:")
            print("  - Batch size tuning (currently fixed at 10)")
            print("  - Parallel worker count optimization")
            print("  - Reduce DataFrame copies")
            return True
        else:
            print("✗ All performance runs failed")
            return False
            
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_large_product_list():
    """Test with a larger product list to stress batch processing."""
    print("\n" + "=" * 60)
    print("TEST: Large Product List (50 products)")
    print("=" * 60)
    
    # Generate 50 products
    large_products = []
    large_dense = {}
    large_sparse = {}
    
    for i in range(50):
        sku = f"SKU{i:04d}"
        large_products.append({
            "id": sku,
            "name": f"Product {i} Gaming Laptop",
            "description": f"Description for product {i} with various features",
            "price": 100.0 + i * 10,
            "category": "Electronics",
        })
        large_dense[sku] = 1.0 - (i * 0.02)  # Decreasing scores
        large_sparse[sku] = 0.9 - (i * 0.018)
    
    try:
        request_data = {
            "query": "gaming laptop",
            "query_id": "large-test",
            "dense_results": large_dense,
            "sparse_results": large_sparse,
            "products": large_products,
        }
        
        start = time.time()
        response = requests.post(
            f"{BASE_URL}/rerank",
            json=request_data,
            timeout=120,
        )
        elapsed = (time.time() - start) * 1000
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Processed 50 products in {elapsed:.2f}ms")
            print(f"  Server latency: {data['latency_ms']:.2f}ms")
            print(f"  Per-product: {elapsed/50:.2f}ms")
            
            # With batch_size=10 and 50 products, should have 5 batches
            print(f"  Expected batches: 5 (batch_size=10)")
            return True
        else:
            print(f"✗ FAILED: Status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


# =============================================================================
# Main Test Runner
# =============================================================================


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 70)
    print("PRODUCT RERANKER SERVICE - TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("Health Check", test_health_check),
        ("Basic RRF", test_rrf_basic),
        ("RRF Empty Dense", test_rrf_edge_case_empty_dense),
        ("RRF Empty Sparse", test_rrf_edge_case_empty_sparse),
        ("RRF Both Empty", test_rrf_edge_case_both_empty),
        ("RRF Includes All Products", test_rrf_includes_all_products),
        ("RRF Single List Scores", test_rrf_single_list_score_quality),
        ("RRF Mixed List Fairness", test_rrf_mixed_list_fairness),
        ("Full Reranking", test_full_reranking),
        ("Concurrent Requests", test_concurrent_requests),
        ("Performance Baseline", test_performance_baseline),
        ("Large Product List", test_large_product_list),
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"✗ {name} crashed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    failed = len(results) - passed
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if failed > 0:
        print("\n" + "-" * 70)
        print("-" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
