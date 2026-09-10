"""Tests for the map-reduce chain."""

from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter
from typing_extensions import override

from langchain_classic.chains.combine_documents.base import (
    BaseCombineDocumentsChain,
)
from langchain_classic.chains.mapreduce import MapReduceChain


class _SingleChunkTextSplitter(TextSplitter):
    @override
    def split_text(self, text: str) -> list[str]:
        return [text]


class _FakeCombineDocumentsChain(BaseCombineDocumentsChain):
    @override
    def combine_docs(
        self,
        docs: list[Document],
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        assert kwargs["metadata"] == "keep"
        return docs[0].page_content, {}

    @override
    async def acombine_docs(
        self,
        docs: list[Document],
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        return self.combine_docs(docs, **kwargs)


def test_invoke_does_not_mutate_inputs() -> None:
    chain = MapReduceChain(
        combine_documents_chain=_FakeCombineDocumentsChain(),
        text_splitter=_SingleChunkTextSplitter(),
    )
    inputs = {"input_text": "hello", "metadata": "keep"}

    result = chain.invoke(inputs)

    assert inputs == {"input_text": "hello", "metadata": "keep"}
    assert result == {
        "input_text": "hello",
        "metadata": "keep",
        "output_text": "hello",
    }
