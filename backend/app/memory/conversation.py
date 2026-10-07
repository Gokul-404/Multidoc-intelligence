"""
Conversation memory for SentinelRAG.
Stores conversation history in Redis with TTL.
Resolves pronouns and context from prior turns without injecting
the full conversation into every retrieval query.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from app.cache.redis_client import get_redis_client
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class ConversationTurn:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class ConversationMemory:
    conversation_id: str
    turns: list[ConversationTurn] = field(default_factory=list)

    def get_context_string(self, max_turns: int = 6) -> str:
        """Return the last N turns as a formatted context string."""
        recent = self.turns[-max_turns:]
        parts = []
        for turn in recent:
            parts.append(f"{turn.role.upper()}: {turn.content}")
        return "\n".join(parts)

    def get_standalone_question(self, question: str) -> str:
        """
        Build a standalone version of the question using recent context.
        Only used for retrieval query improvement (not full history injection).
        """
        if not self.turns:
            return question
        # Take the last assistant response and user's follow-up
        context_parts = []
        for turn in self.turns[-4:]:
            context_parts.append(f"{turn.role}: {turn.content[:200]}")
        context = " | ".join(context_parts)
        return f"{question} (context: {context})"


class ConversationManager:
    """Redis-backed conversation state manager."""

    def __init__(self) -> None:
        self._redis = get_redis_client()

    def _key(self, conversation_id: str, tenant_id: str) -> str:
        return f"conv:{tenant_id}:{conversation_id}"

    async def get(self, conversation_id: str, tenant_id: str) -> ConversationMemory:
        """Load conversation from Redis."""
        key = self._key(conversation_id, tenant_id)
        memory = ConversationMemory(conversation_id=conversation_id)
        try:
            raw = await self._redis.get(key)
            if raw:
                data = json.loads(raw)
                memory.turns = [ConversationTurn(**t) for t in data.get("turns", [])]
        except Exception as exc:
            logger.warning("Failed to load conversation %s: %s", conversation_id, exc)
        return memory

    async def save(self, memory: ConversationMemory, tenant_id: str) -> None:
        """Save conversation to Redis."""
        key = self._key(memory.conversation_id, tenant_id)
        try:
            data = {"turns": [{"role": t.role, "content": t.content} for t in memory.turns]}
            await self._redis.setex(
                key,
                settings.redis_conversation_ttl,
                json.dumps(data),
            )
        except Exception as exc:
            logger.warning("Failed to save conversation: %s", exc)

    async def add_turn(
        self,
        conversation_id: str,
        tenant_id: str,
        user_message: str,
        assistant_message: str,
    ) -> ConversationMemory:
        """Append a user + assistant turn pair."""
        memory = await self.get(conversation_id, tenant_id)
        memory.turns.append(ConversationTurn(role="user", content=user_message))
        memory.turns.append(ConversationTurn(role="assistant", content=assistant_message))
        # Keep last 20 turns to prevent unbounded growth
        memory.turns = memory.turns[-20:]
        await self.save(memory, tenant_id)
        return memory
