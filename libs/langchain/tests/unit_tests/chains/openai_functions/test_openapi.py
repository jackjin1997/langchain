"""Tests for OpenAPI function chains."""

from typing import Any
from unittest.mock import Mock

from requests import Response, codes

from langchain_classic.chains.openai_functions.openapi import SimpleRequestChain


def test_simple_request_chain_does_not_mutate_inputs() -> None:
    response = Mock(spec=Response)
    response.status_code = codes.ok
    response.json.return_value = {"ok": True}
    request = Mock(return_value=response)
    chain = SimpleRequestChain(request_method=request)
    inputs: dict[str, Any] = {
        "function": {
            "name": "get_item",
            "arguments": {"params": {"id": "1"}},
        },
        "trace": "keep",
    }

    result = chain.invoke(inputs)

    request.assert_called_once_with("get_item", {"params": {"id": "1"}})
    assert inputs == {
        "function": {
            "name": "get_item",
            "arguments": {"params": {"id": "1"}},
        },
        "trace": "keep",
    }
    assert result == {**inputs, "response": {"ok": True}}
