import httpx
import json
import asyncio
import textwrap

QUERIES = [
    ("Vertical SaaS for logistics operators in the UK", "long_offering describes software product + logistics domain + UK market"),
    ("Industrial software providers with field-deployed delivery", "long_offering signals on-premise or field deployment, not cloud-only"),
    ("Companies solving onboarding inefficiency for frontline teams", "Problem + use-case signal present in long_offering, not just category tags"),
    ("Fintech companies not focused on payments", "Exclusion intent respected - payments-heavy long_offering should rank down or drop"),
    ("a", "Graceful fallback - must not crash, must return a sensible default response")
]

async def test_all():
    async with httpx.AsyncClient(timeout=60.0) as client:
        for idx, (q, criteria) in enumerate(QUERIES, 1):
            print(f"\n{'='*80}\nQuery {idx}: '{q}'")
            print(f"Goal: {criteria}")
            
            try:
                # 1. Run Agent Refine
                refine_req = await client.post("http://127.0.0.1:8000/agent/refine", json={
                    "message": q, "max_iterations": 3
                })
                refine_req.raise_for_status()
                agent_data = refine_req.json()
                refined_query = agent_data["refined_query"]
                print(f"\n[Agent] Iterations: {agent_data['iterations_used']}")
                print(f"[Agent] Refined Query: {refined_query}")
                print(f"[Agent] Rationale: {agent_data['rationale']}")
                
                # 2. Run Search
                search_req = await client.post("http://127.0.0.1:8000/search/run", json={
                    "query": refined_query,
                    "top_k_raw": 1000,
                    "top_k_final": 3
                })
                search_req.raise_for_status()
                search_data = search_req.json()
                
                print(f"\n[Results] Total found: {search_data['total']}")
                for i, r in enumerate(search_data["results"][:2], 1):
                    bio = textwrap.shorten(r['long_offering'], width=150, placeholder="...")
                    print(f"  {i}. {r['company_name']} ({r['country']}) - Score: {r['score']}")
                    print(f"     Bio: {bio}")
                    
            except Exception as e:
                print(f"Error testing query: {e}")

if __name__ == "__main__":
    asyncio.run(test_all())
