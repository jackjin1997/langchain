"""Test async save behavior for classic memory implementations."""

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import FakeListLLM
from langchain_core.messages import BaseMessage

from langchain_classic.memory.entity import (
    ConversationEntityMemory,
    InMemoryEntityStore,
)
from langchain_classic.memory.summary import ConversationSummaryMemory
from langchain_classic.memory.token_buffer import ConversationTokenBufferMemory


class CountingLLM(FakeListLLM):
    """Fake LLM that counts every message as one token."""

    def get_num_tokens_from_messages(
        self,
        messages: list[BaseMessage],
        tools: Sequence[Any] | None = None,
    ) -> int:
        del tools
        return len(messages)


async def test_token_buffer_prunes_after_async_save() -> None:
    memory = ConversationTokenBufferMemory(
        llm=CountingLLM(responses=["unused"]),
        max_token_limit=1,
    )

    await memory.asave_context({"input": "hello"}, {"output": "world"})

    assert len(memory.chat_memory.messages) == 1


async def test_summary_updates_after_async_save() -> None:
    memory = ConversationSummaryMemory(llm=FakeListLLM(responses=["summary"]))

    await memory.asave_context({"input": "hello"}, {"output": "world"})

    assert memory.buffer == "summary"


async def test_entity_store_updates_after_async_save() -> None:
    entity_store = InMemoryEntityStore(store={"Alice": "old summary"})
    memory = ConversationEntityMemory(
        llm=FakeListLLM(responses=["updated summary"]),
        entity_cache=["Alice"],
        entity_store=entity_store,
    )

    await memory.asave_context({"input": "hello"}, {"output": "world"})

    assert entity_store.get("Alice") == "updated summary"
