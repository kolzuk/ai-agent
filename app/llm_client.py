"""LLM client for interacting with OpenAI and Yandex GPT."""

import json
import logging
from typing import Any, Dict, List, Optional

import aiohttp
import requests
from openai import OpenAI

# Configuration will be passed as parameters instead of importing global config

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for interacting with Language Learning Models."""

    def __init__(
        self,
        use_yandex_gpt: bool = False,
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4",
        yandex_gpt_api_key: Optional[str] = None,
        yandex_gpt_folder_id: Optional[str] = None,
    ) -> None:
        """Initialize the LLM client."""
        self.use_yandex_gpt = use_yandex_gpt
        self.openai_model = openai_model
        self.yandex_gpt_api_key = yandex_gpt_api_key
        self.yandex_gpt_folder_id = yandex_gpt_folder_id
        
        if not self.use_yandex_gpt:
            if not openai_api_key:
                raise ValueError("OpenAI API key is required when not using Yandex GPT")
            try:
                # Try to create OpenAI client with minimal parameters to avoid compatibility issues
                self.openai_client = OpenAI(api_key=openai_api_key)
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                # Fallback: set to None and handle gracefully
                self.openai_client = None
        else:
            if not yandex_gpt_api_key or not yandex_gpt_folder_id:
                raise ValueError("Yandex GPT API key and folder ID are required when using Yandex GPT")
            self.openai_client = None

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 4000,
        temperature: float = 0.7,
    ) -> str:
        """Generate a response using the configured LLM."""
        if self.use_yandex_gpt:
            return await self._generate_yandex_response(
                messages, max_tokens, temperature
            )
        else:
            return self._generate_openai_response(messages, max_tokens, temperature)

    def _generate_openai_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate response using OpenAI API."""
        if not self.openai_client:
            raise ValueError("OpenAI client is not initialized")
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Error generating OpenAI response: {e}")
            raise

    async def _generate_yandex_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate response using Yandex GPT API."""
        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.yandex_gpt_api_key}",
        }

        # Convert OpenAI format messages to Yandex format
        yandex_messages = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "assistant"
            yandex_messages.append({"role": role, "text": msg["content"]})

        data = {
            "modelUri": f"gpt://{self.yandex_gpt_folder_id}/yandexgpt-lite",
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": str(max_tokens),
            },
            "messages": yandex_messages,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["result"]["alternatives"][0]["message"]["text"]
                    else:
                        error_text = await response.text()
                        logger.error(f"Yandex GPT API error: {error_text}")
                        raise Exception(f"Yandex GPT API error: {error_text}")
        except Exception as e:
            logger.error(f"Error generating Yandex GPT response: {e}")
            raise

    def create_system_message(self, content: str) -> Dict[str, str]:
        """Create a system message."""
        return {"role": "system", "content": content}

    def create_user_message(self, content: str) -> Dict[str, str]:
        """Create a user message."""
        return {"role": "user", "content": content}

    def create_assistant_message(self, content: str) -> Dict[str, str]:
        """Create an assistant message."""
        return {"role": "assistant", "content": content}