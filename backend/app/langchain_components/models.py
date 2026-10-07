"""
LangChain model abstraction for SentinelRAG.
Wraps Google Gemini 2.0 Flash with structured output support.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any, Optional, Type, TypeVar

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from app.config import get_settings

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class ModelService:
    """
    Centralised model abstraction using LangChain's ChatGoogleGenerativeAI.
    Provides both raw generation and structured output via with_structured_output.
    """

    def __init__(self, model_name: Optional[str] = None, temperature: Optional[float] = None) -> None:
        settings = get_settings()
        self._model_name = model_name or settings.gemini_model
        self._temperature = temperature if temperature is not None else settings.temperature
        self._llm = ChatGoogleGenerativeAI(
            model=self._model_name,
            google_api_key=settings.google_api_key,
            temperature=self._temperature,
            max_output_tokens=settings.max_answer_tokens,
            convert_system_message_to_human=False,
        )
        logger.info("ModelService initialised with model=%s temp=%.1f", self._model_name, self._temperature)

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    def generate(self, prompt_value: Any) -> str:
        """Invoke LLM and return raw text content."""
        response: AIMessage = self._llm.invoke(prompt_value)
        return response.content  # type: ignore[return-value]

    def generate_structured(self, prompt_value: Any, schema: Type[T]) -> T:
        """Invoke LLM with structured output and return typed Pydantic object."""
        structured_llm = self._llm.with_structured_output(schema)
        return structured_llm.invoke(prompt_value)  # type: ignore[return-value]

    def parse_json_response(self, raw: str) -> Any:
        """
        Safely parse a JSON response that may be wrapped in markdown code blocks.
        Never trusts the document content path; only used for model outputs.
        """
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last fence lines
            inner = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
            text = inner.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response: %.200s", text)
            return None

    async def agenerate(self, prompt_value: Any) -> str:
        """Async invoke."""
        response = await self._llm.ainvoke(prompt_value)
        return response.content  # type: ignore[return-value]


@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    return ModelService()
