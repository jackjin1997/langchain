from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import pytest
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models

from langchain_qdrant import QdrantVectorStore, RetrievalMode
from tests.integration_tests.common import (
    ConsistentFakeEmbeddings,
    ConsistentFakeSparseEmbeddings,
    assert_documents_equals,
)
from tests.integration_tests.fixtures import qdrant_locations, retrieval_modes

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
@pytest.mark.parametrize("retrieval_mode", retrieval_modes())
@pytest.mark.parametrize(
    "sparse_vector_name", ["my-sparse-vector", "another-sparse-vector"]
)
def test_qdrant_add_documents_extends_existing_collection(
    location: str,
    vector_name: str,
    retrieval_mode: RetrievalMode,
    sparse_vector_name: str,
) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = QdrantVectorStore.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        location=location,
        vector_name=vector_name,
        retrieval_mode=retrieval_mode,
        sparse_vector_name=sparse_vector_name,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
    )

    new_texts = ["foobar", "foobaz"]
    docsearch.add_documents([Document(page_content=content) for content in new_texts])
    output = docsearch.similarity_search("foobar", k=1)
    assert_documents_equals(output, [Document(page_content="foobar")])


@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
@pytest.mark.parametrize("retrieval_mode", retrieval_modes())
@pytest.mark.parametrize(
    "sparse_vector_name", ["my-sparse-vector", "another-sparse-vector"]
)
@pytest.mark.parametrize("batch_size", [1, 64])
def test_qdrant_add_texts_returns_all_ids(
    location: str,
    vector_name: str,
    retrieval_mode: RetrievalMode,
    sparse_vector_name: str,
    batch_size: int,
) -> None:
    """Test end to end Qdrant.add_texts returns unique ids."""
    docsearch = QdrantVectorStore.from_texts(
        ["foobar"],
        ConsistentFakeEmbeddings(),
        location=location,
        vector_name=vector_name,
        retrieval_mode=retrieval_mode,
        sparse_vector_name=sparse_vector_name,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
        batch_size=batch_size,
    )

    ids = docsearch.add_texts(["foo", "bar", "baz"])
    assert len(ids) == 3
    assert len(set(ids)) == 3
    assert len(docsearch.get_by_ids(ids)) == 3


@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
def test_qdrant_add_texts_stores_duplicated_texts(
    location: str,
    vector_name: str,
) -> None:
    """Test end to end Qdrant.add_texts stores duplicated texts separately."""
    client = QdrantClient(location)
    collection_name = uuid.uuid4().hex
    vectors_config = {
        vector_name: models.VectorParams(size=10, distance=models.Distance.COSINE)
    }
    client.recreate_collection(collection_name, vectors_config=vectors_config)

    vec_store = QdrantVectorStore(
        client,
        collection_name,
        embedding=ConsistentFakeEmbeddings(),
        vector_name=vector_name,
    )
    ids = vec_store.add_texts(["abc", "abc"], [{"a": 1}, {"a": 2}])

    assert len(set(ids)) == 2
    assert client.count(collection_name).count == 2


@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
@pytest.mark.parametrize("retrieval_mode", retrieval_modes())
@pytest.mark.parametrize(
    "sparse_vector_name", ["my-sparse-vector", "another-sparse-vector"]
)
@pytest.mark.parametrize("batch_size", [1, 64])
def test_qdrant_add_texts_stores_ids(
    location: str,
    vector_name: str,
    retrieval_mode: RetrievalMode,
    sparse_vector_name: str,
    batch_size: int,
) -> None:
    """Test end to end Qdrant.add_texts stores provided ids."""
    ids: list[str | int] = [
        "fa38d572-4c31-4579-aedc-1960d79df6df",
        432,
        432145435,
    ]
    collection_name = uuid.uuid4().hex
    vec_store = QdrantVectorStore.from_texts(
        ["abc", "def", "ghi"],
        ConsistentFakeEmbeddings(),
        ids=ids,
        collection_name=collection_name,
        location=location,
        vector_name=vector_name,
        retrieval_mode=retrieval_mode,
        sparse_vector_name=sparse_vector_name,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
        batch_size=batch_size,
    )

    assert vec_store.client.count(collection_name).count == 3
    stored_ids = [point.id for point in vec_store.client.scroll(collection_name)[0]]
    assert set(ids) == set(stored_ids)
    assert len(vec_store.get_by_ids(ids)) == 3


@pytest.mark.parametrize(
    "ids",
    [
        [],
        ["fa38d572-4c31-4579-aedc-1960d79df6df"],
        [
            "fa38d572-4c31-4579-aedc-1960d79df6df",
            "9f587830-1468-4f44-9e5a-2df797b2978a",
            "a7e6395c-8d0c-498c-8f79-e301e925a62c",
        ],
    ],
)
def test_qdrant_add_texts_rejects_mismatched_ids(
    ids: list[str], mocker: MockerFixture
) -> None:
    """Test that Qdrant.add_texts rejects mismatched text and ID counts."""
    vec_store = QdrantVectorStore.from_texts(
        ["seed"],
        ConsistentFakeEmbeddings(),
        location=":memory:",
    )
    count_before = vec_store.client.count(vec_store.collection_name).count
    build_vectors = mocker.spy(vec_store, "_build_vectors")

    with pytest.raises(ValueError, match="number of ids must match"):
        vec_store.add_texts(["document A", "document B"], ids=ids)

    build_vectors.assert_not_called()
    assert vec_store.client.count(vec_store.collection_name).count == count_before
