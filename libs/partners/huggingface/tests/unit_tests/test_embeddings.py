"""Unit tests for Hugging Face embeddings."""

from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from langchain_huggingface.embeddings import HuggingFaceEmbeddings


def test_multi_process_pool_stopped_when_encoding_fails() -> None:
    """Test worker cleanup when multi-process encoding raises."""
    pool = object()
    stop_multi_process_pool = Mock()
    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = SimpleNamespace(  # type: ignore[attr-defined]
        stop_multi_process_pool=stop_multi_process_pool
    )
    client = Mock()
    client.start_multi_process_pool.return_value = pool
    client.encode_multi_process.side_effect = RuntimeError("encoding failed")
    embeddings = HuggingFaceEmbeddings.model_construct(
        model_name="fake",
        cache_folder=None,
        model_kwargs={},
        encode_kwargs={},
        query_encode_kwargs={},
        multi_process=True,
        show_progress=False,
    )
    object.__setattr__(embeddings, "_client", client)

    with (
        patch.dict("sys.modules", {"sentence_transformers": sentence_transformers}),
        pytest.raises(RuntimeError, match="encoding failed"),
    ):
        embeddings._embed(["text"], {})

    stop_multi_process_pool.assert_called_once_with(pool)