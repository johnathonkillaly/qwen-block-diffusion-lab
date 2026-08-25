"""Thin OpenAI-compatible client for LM Studio and OMLX.

SCOPE LIMIT, ON PURPOSE. These servers expose `/v1/completions` and
`/v1/chat/completions`. Diffusion decoding needs to feed an arbitrary noisy token
canvas through the model and read logits at *every* canvas position -- an
operation no OpenAI-compatible endpoint exposes. So these clients are used for
autoregressive baselines only, and `diffusion_generate` never touches them.

Faking diffusion through a completions endpoint (e.g. by round-tripping text) would
produce numbers that look like results and are not.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .ar_baseline import ARBaselineResult

LMSTUDIO_DEFAULT = "http://localhost:1234/v1"
OMLX_DEFAULT = "http://localhost:8080/v1"


@dataclass
class OpenAICompatClient:
    base_url: str = LMSTUDIO_DEFAULT
    name: str = "lmstudio"
    timeout: float = 120.0

    def _request(self, path: str, payload: dict | None = None) -> dict:
        url = f"{self.base_url.rstrip('/')}{path}"
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def is_up(self) -> bool:
        try:
            self._request("/models")
            return True
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            return False

    def list_models(self) -> list[str]:
        try:
            return [m["id"] for m in self._request("/models").get("data", [])]
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, json.JSONDecodeError):
            return []

    def complete(
        self, prompt: str, model: str, max_new_tokens: int = 64, temperature: float = 0.0
    ) -> ARBaselineResult:
        t0 = time.time()
        resp = self._request(
            "/completions",
            {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "stream": False,
            },
        )
        elapsed = time.time() - t0
        text = resp["choices"][0].get("text", "")
        n = resp.get("usage", {}).get("completion_tokens", 0) or len(text.split())
        return ARBaselineResult(
            backend=self.name,
            prompt=prompt,
            output_text=text,
            output_tokens=int(n),
            seconds=elapsed,
            tokens_per_sec=n / elapsed if elapsed else 0.0,
            memory_kind="server-side (not measurable from here)",
            notes=[
                "OpenAI-compatible endpoint: AR baseline only. Cannot run diffusion "
                "decoding, which requires per-position logits over a rewritable canvas."
            ],
        )


def lmstudio_client(base_url: str = LMSTUDIO_DEFAULT) -> OpenAICompatClient:
    return OpenAICompatClient(base_url=base_url, name="lmstudio")


def omlx_client(base_url: str = OMLX_DEFAULT) -> OpenAICompatClient:
    return OpenAICompatClient(base_url=base_url, name="omlx")
