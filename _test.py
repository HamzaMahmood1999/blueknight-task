import httpx
import json
import asyncio

async def test_search():
    print("Testing /search/run...")
    payload = {
        "query": {
            "query_text": "SaaS for HR and payroll",
            "geography": ["United States"],
            "exclusions": ["finance"]
        },
        "top_k_raw": 100,
        "top_k_final": 5
    }
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            req = await client.post("http://127.0.0.1:8000/search/run", json=payload)
            req.raise_for_status()
            print("Search Response:", json.dumps(req.json(), indent=2))
        except Exception as e:
            print(f"Search endpoint error: {e}")

async def test_refine():
    print("\nTesting /agent/refine...")
    payload = {
        "message": "SaaS for HR and payroll in US, no finance tools",
        "max_iterations": 3
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            req = await client.post("http://127.0.0.1:8000/agent/refine", json=payload)
            req.raise_for_status()
            print("Refine Response:", json.dumps(req.json(), indent=2))
        except Exception as e:
            print(f"Refine endpoint error: {e}")

if __name__ == "__main__":
    asyncio.run(test_search())
    asyncio.run(test_refine())
