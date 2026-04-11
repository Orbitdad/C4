from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .llm_client import LLMClient


@dataclass(frozen=True)
class RoleSpec:
    model: str
    max_tokens: Optional[int] = None
    options: Optional[Dict[str, Any]] = None


class RoleBasedLLM:
    """
    Role-based multi-model orchestration over a single provider (typically Ollama).

    Exposes the same interface C4 already uses:
    - generate(prompt, system_message, history)
    - embed(text)

    Plus a role-based call:
    - call_llm(role=..., ...)
    """

    def __init__(
        self,
        base_client: LLMClient,
        roles: Dict[str, RoleSpec],
        embed_model: Optional[str] = None,
        default_role: str = "explainer",
    ) -> None:
        self._base = base_client
        self._roles = roles
        self._embed_model = embed_model
        self._default_role = default_role

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RoleBasedLLM":
        llm_cfg = config.get("llm", {}) or {}

        base = LLMClient.from_config(config)

        # Backwards-compatible: if roles not present, treat the configured model as all roles.
        roles_cfg = llm_cfg.get("roles") or {}
        if not roles_cfg:
            model = llm_cfg.get("model", base.model)
            roles_cfg = {
                "planner": {"model": model},
                "coder": {"model": model},
                "debugger": {"model": model},
                "explainer": {"model": model},
            }

        roles: Dict[str, RoleSpec] = {}
        for role_name, rc in roles_cfg.items():
            if not isinstance(rc, dict) or not rc.get("model"):
                continue
            roles[role_name] = RoleSpec(
                model=str(rc["model"]),
                max_tokens=int(rc["max_tokens"]) if rc.get("max_tokens") is not None else None,
                options=rc.get("options") if isinstance(rc.get("options"), dict) else None,
            )

        embed_model = llm_cfg.get("embed_model") or llm_cfg.get("embedding_model")
        default_role = str(llm_cfg.get("default_role") or "explainer")

        return cls(base_client=base, roles=roles, embed_model=embed_model, default_role=default_role)

    # --- Compatibility methods used by existing C4 code ---

    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        return self.call_llm(
            role=self._default_role,
            prompt=prompt,
            system_message=system_message,
            history=history,
        )

    def embed(self, text: str) -> List[float]:
        return self._base.embed(text, model=self._embed_model)

    # --- Role-based orchestration ---

    def call_llm(
        self,
        role: str,
        prompt: str,
        system_message: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        model_override: Optional[str] = None,
        options_override: Optional[Dict[str, Any]] = None,
        provider_override: Optional[str] = None,
    ) -> str:
        # Respect the caller's role exactly — no keyword hijacking.
        # provider_override is only applied when the caller explicitly passes it
        # (e.g., the debug loop escalating to Gemini at loop >= 2).
        effective_provider = provider_override  # None → use role's configured provider

        spec = self._roles.get(role) or self._roles.get(self._default_role)
        model = model_override or (spec.model if spec else None)

        options: Dict[str, Any] = {}
        if spec and spec.options:
            options.update(spec.options)
        if options_override:
            options.update({k: v for k, v in options_override.items() if v is not None})

        # Note: max_tokens support is provider-specific; current LLMClient uses self.max_tokens for Gemini.
        # For Ollama we pass through options (e.g., num_ctx) and rely on model behavior.
        return self._base.generate(
            prompt=prompt,
            system_message=system_message,
            history=history,
            model=model,
            options=options if options else None,
            provider_override=effective_provider,
        )

