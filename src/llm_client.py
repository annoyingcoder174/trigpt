# src/llm_client.py

import os
import requests
from typing import List, Dict, Any


class LLMClient:
    """
    Simple client to talk to a local Ollama model via HTTP.
    """

    def __init__(self, model_name: str | None = None, base_url: str = "http://localhost:11434"):
        # You can override the model with an env var if you want
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3")
        self.base_url = base_url.rstrip("/")

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        messages example:
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]
        """
        url = f"{self.base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
        }

        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        # Ollama returns a JSON with `message` and `content`
        message = data.get("message", {})
        content = message.get("content", "")
        return content.strip()
