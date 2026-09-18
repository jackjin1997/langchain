"""Test async clear behavior for classic memory implementations."""

from langchain_core.language_models import FakeListLLM
from langchain_core.messages import HumanMessage

from langchain_classic.memory.entity import (
    ConversationEntityMemory,
    InMemoryEntityStore,
)
from langchain_classic.memory.summary import ConversationSummaryMemory


async def test_summary_async_clear_resets_buffer() -> None:
    memory = ConversationSummaryMemory(llm=FakeListLLM(responses=["unused"]))
    memory.buffer = "stale summary"
    memory.chat_memory.add_message(HumanMessage(content="hello"))

    await memory.aclear()

    assert memory.buffer == ""
    assert memory.chat_memory.messages == []


async def test_entity_async_clear_resets_derived_state() -> None:
    entity_store = InMemoryEntityStore(store={"Alice": "summary"})
    memory = ConversationEntityMemory(
        llm=FakeListLLM(responses=["unused"]),
        entity_cache=["Alice"],
        entity_store=entity_store,
    )
    memory.chat_memory.add_message(HumanMessage(content="hello"))

    await memory.aclear()

    assert memory.chat_memory.messages == []
    assert memory.entity_cache == []
    assert entity_store.get("Alice") is None
