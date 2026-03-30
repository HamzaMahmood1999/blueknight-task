"""Groq LLM client for structured query refinement."""
from __future__ import annotations

import json
import logging
from typing import Any

from groq import AsyncGroq

from app import config
from app.utils.json_contract import parse_json_contract

logger = logging.getLogger("blueknight")

# System prompt instructing the LLM to produce structured query refinements
REFINER_SYSTEM_PROMPT = """You are an expert M&A search query refiner. Your job is to take a user's
natural language company search request and produce a structured search query.

You must output ONLY valid JSON with exactly these fields:
{
  "query_text": "optimized search text for vector similarity matching against company bios",
  "geography": ["list", "of", "target", "countries/regions"],
  "exclusions": ["terms", "to", "exclude", "from", "results"],
  "reasoning": "brief explanation of your refinement choices"
}

Guidelines:
- query_text should be a rich semantic description that would match well against company long_offering bios
- Extract geography signals (countries, regions) into the geography list
- Extract exclusion/negative signals into the exclusions list
- If the user says "not X" or "excluding X", put X in exclusions
- If no geography is specified, leave geography as []
- If no exclusions, leave exclusions as []
- Expand abbreviations (UK -> United Kingdom, US -> United States)
- For vague or short queries, produce a reasonable default interpretation
- If given previous results and diagnostics, refine the query to improve results

Always respond with valid JSON only. No markdown, no explanation outside the JSON."""


class LLMClient:
    """Thin wrapper around Groq API for structured query refinement."""

    def __init__(self) -> None:
        self._client = AsyncGroq(api_key=config.GROQ_API_KEY)

    async def refine_query(
        self,
        user_message: str,
        history: list[dict[str, Any]] | None = None,
        previous_results_summary: str | None = None,
        previous_query: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call Groq to produce a refined structured query.

        Returns:
            Parsed dict with query_text, geography, exclusions, reasoning.
            On failure, returns a fallback dict with the user message as query_text.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": REFINER_SYSTEM_PROMPT}
        ]

        if history:
            # We skip adding actual history role messages to preserve the strict JSON output constraint
            # Instead we inject it into the prompt context
            history_str = json.dumps(history[-5:])  # Last 5 turns
            prompt_context = f"Conversation history:\n{history_str}\n\n"
        else:
            prompt_context = ""

        # Build the current turn user message
        user_content = f"User search request: {user_message}"

        if previous_query:
            user_content += f"\n\nPrevious query that was tried:\n{json.dumps(previous_query)}"

        if previous_results_summary:
            user_content += f"\n\nResults from previous query:\n{previous_results_summary}"
            user_content += "\n\nPlease refine the query to improve the results. Consider adjusting query_text, geography, or exclusions."

        messages.append({"role": "user", "content": prompt_context + user_content})

        try:
            response = await self._client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )

            raw_text = response.choices[0].message.content or ""
            parsed = parse_json_contract(raw_text)

            # Track token usage if available
            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            parsed["_usage"] = usage
            return parsed

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return {
                "query_text": user_message,
                "geography": [],
                "exclusions": [],
                "reasoning": f"LLM fallback — using raw user message. Error: {str(e)[:100]}",
                "_usage": {},
                "_error": str(e),
            }
