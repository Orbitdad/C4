"""
LLM client wrapper for C4/JARVIS reasoning.

Supports:
- Gemini (single model)
- Ollama (single model or per-call model override)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from jarvis.config_loader import get_llm_api_key


class LLMClient:
    """
    Abstraction over Gemini, Ollama, or other LLM APIs.
    """

    def __init__(
        self,
        api_key: str = "",
        provider: str = "gemini",
        model: str = "gemini-1.5-flash",
        max_tokens: int = 2048,
        ollama_host: str = "http://localhost:11434",
    ) -> None:
        self.api_key = api_key
        self.provider = provider.lower()
        self.model = model
        self.max_tokens = max_tokens
        self.ollama_host = ollama_host
        self._genai = None
        self._ollama_client = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LLMClient":
        llm_cfg = config.get("llm", {}) or {}
        provider = llm_cfg.get("provider", "gemini")
        
        # Get Gemini API key if needed
        api_key = get_llm_api_key(config) if provider == "gemini" else ""
        
        return cls(
            api_key=api_key or llm_cfg.get("api_key", ""),
            provider=provider,
            model=llm_cfg.get("model", "gemini-1.5-flash"),
            max_tokens=int(llm_cfg.get("max_tokens") or 2048),
            ollama_host=llm_cfg.get("ollama_host", "http://localhost:11434"),
        )

    def _get_gemini_client(self):
        if self._genai is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._genai = genai
        return self._genai

    def _get_ollama_client(self):
        if self._ollama_client is None:
            import ollama
            self._ollama_client = ollama.Client(host=self.ollama_host)
        return self._ollama_client

    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        provider_override: Optional[str] = None,
    ) -> str:
        """Generate a completion using the configured provider."""
        effective_provider = provider_override or self.provider
        if effective_provider == "ollama":
            return self._generate_ollama(prompt, system_message, history, model=model, options=options)
        return self._generate_gemini(prompt, system_message, history, model=model)

    def embed(self, text: str, model: Optional[str] = None) -> List[float]:
        """Convert text into a 1D vector float array strictly for Semantic matching."""
        if not text:
            return []
        if self.provider == "ollama":
            try:
                client = self._get_ollama_client()
                response = client.embeddings(model=model or self.model, prompt=text)
                return response.get("embedding", [])
            except Exception as e:
                # Silently degrade — model may not be installed yet
                err = str(e).lower()
                if "not found" not in err and "404" not in err:
                    logger.debug(f"[Embed] Ollama error: {e}")
                return []
        else:
            try:
                genai = self._get_gemini_client()
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text
                )
                return result.get("embedding", [])
            except Exception:
                return []


    def _generate_gemini(
        self,
        prompt: str,
        system_message: str | None,
        history: list[dict] | None,
        model: Optional[str] = None,
    ) -> str:
        if not self.api_key or self.api_key == "dummy":
            return ""
        try:
            genai = self._get_gemini_client()
            mdl = genai.GenerativeModel(model or self.model)
            full_prompt = (system_message + "\n\n" + prompt) if system_message else prompt
            
            if history:
                chat = mdl.start_chat(history=history)
                response = chat.send_message(full_prompt)
            else:
                response = mdl.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(max_output_tokens=self.max_tokens),
                )
            return response.text.strip() if response and response.text else ""
        except Exception as e:
            return f"[Error: {e}]"

    def _generate_ollama(
        self,
        prompt: str,
        system_message: str | None,
        history: list[dict] | None,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        try:
            client = self._get_ollama_client()
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": prompt})

            # Optimize for near-zero latency
            base_options: Dict[str, Any] = {
                "temperature": 0.2,
                "top_k": 40,
                "top_p": 0.9,
                "num_ctx": 2048,
                "num_predict": self.max_tokens # Map max_tokens to num_predict
            }
            if options:
                base_options.update({k: v for k, v in options.items() if v is not None})

            response = client.chat(model=(model or self.model), messages=messages, options=base_options)
            return response["message"]["content"].strip()
        except Exception as e:
            return f"[Error: {e}]"
