"""A single, isolated point of access to the model.

Main points:
I keep model choice + credentials in ONE module so the rest of the
codebase never touches an SDK directly. Swapping OpenAI for a local vLLM/Ollama
endpoint, or routing through a gateway (Portkey, in Thematica), is a one-line change
here and nowhere else. The base_url swap is exactly the "OpenAI-compatible runtime"
pattern I used to evaluate open-source models.
"""

import os
from openai import OpenAI


class LLMClient:
    def __init__(self, model: str | None = None):
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "not-needed-for-local"),
            # Unset -> real OpenAI. Set OPENAI_BASE_URL to point at Ollama/vLLM/gateway.
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
        )
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        """One chat completion. temperature is a knob the caller sets deliberately:
        higher for exploration (we want diverse questions), 0 for grounding/summarizing
        (we want reproducibility)."""
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""
