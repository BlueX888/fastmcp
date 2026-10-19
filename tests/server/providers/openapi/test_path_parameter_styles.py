"""Path parameters are serialized according to their declared OpenAPI style."""

from typing import Any

import httpx2
import pytest

from fastmcp import Client, FastMCP


def _spec(paths: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "openapi": "3.1.0",
        "info": {"title": "Path parameter styles", "version": "1"},
        "paths": paths,
    }


def _operation(operation_id: str, parameter: dict[str, Any]) -> dict[str, Any]:
    return {
        "operationId": operation_id,
        "parameters": [parameter],
        "responses": {"200": {"description": "OK"}},
    }


async def _capture_request(
    paths: dict[str, dict[str, Any]],
    tool_name: str,
    arguments: dict[str, Any],
) -> httpx2.Request:
    requests: list[httpx2.Request] = []

    def capture(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json={"ok": True})

    async with httpx2.AsyncClient(
        base_url="https://example.test", transport=httpx2.MockTransport(capture)
    ) as http_client:
        server = FastMCP.from_openapi(openapi_spec=_spec(paths), client=http_client)
        async with Client(server) as client:
            await client.call_tool(tool_name, arguments)

    assert len(requests) == 1
    return requests[0]


@pytest.mark.parametrize(
    "style,explode,value,expected_path",
    [
        # simple is the default and keeps its existing single-segment behavior.
        (None, None, "abc", "/items/abc"),
        ("simple", None, "abc", "/items/abc"),
        # label scalars are prefixed with "."; matrix scalars use ";name=".
        ("label", None, "abc", "/items/.abc"),
        ("matrix", None, "abc", "/items/;id=abc"),
        ("label", False, "abc", "/items/.abc"),
        ("matrix", False, "abc", "/items/;id=abc"),
        # Exploded arrays join with "." for label and repeat ";name=" for matrix.
        ("label", True, ["a", "b"], "/items/.a.b"),
        ("matrix", True, ["a", "b"], "/items/;id=a;id=b"),
        # Non-exploded arrays use a comma in both styles.
        ("label", False, ["a", "b"], "/items/.a,b"),
        ("matrix", False, ["a", "b"], "/items/;id=a,b"),
        # Exploded objects use "key=value" pairs.
        ("label", True, {"R": 100, "G": 200}, "/items/.R=100.G=200"),
        ("matrix", True, {"R": 100, "G": 200}, "/items/;R=100;G=200"),
        # Non-exploded objects alternate keys and values with commas.
        ("label", False, {"R": 100}, "/items/.R,100"),
        ("matrix", False, {"R": 100}, "/items/;id=R,100"),
    ],
)
async def test_path_parameter_style(
    style: str | None,
    explode: bool | None,
    value: Any,
    expected_path: str,
) -> None:
    parameter: dict[str, Any] = {
        "name": "id",
        "in": "path",
        "required": True,
        "schema": {"type": "string"},
    }
    if isinstance(value, list):
        parameter["schema"] = {"type": "array", "items": {"type": "string"}}
    elif isinstance(value, dict):
        parameter["schema"] = {
            "type": "object",
            "properties": {key: {"type": "integer"} for key in value},
        }
    if style is not None:
        parameter["style"] = style
    if explode is not None:
        parameter["explode"] = explode

    request = await _capture_request(
        {"/items/{id}": {"get": _operation("get_item", parameter)}},
        "get_item",
        {"id": value},
    )

    assert request.url.path == expected_path


async def test_label_path_parameter_percent_encodes_value() -> None:
    """Structural dots stay literal while value content is encoded."""
    request = await _capture_request(
        {
            "/items/{id}": {
                "get": _operation(
                    "get_item",
                    {
                        "name": "id",
                        "in": "path",
                        "required": True,
                        "style": "label",
                        "schema": {"type": "string"},
                    },
                )
            }
        },
        "get_item",
        {"id": "v1.2/a b"},
    )

    assert request.url.raw_path == b"/items/.v1%2E2%2Fa%20b"


async def test_matrix_path_parameter_percent_encodes_value() -> None:
    request = await _capture_request(
        {
            "/items/{id}": {
                "get": _operation(
                    "get_item",
                    {
                        "name": "id",
                        "in": "path",
                        "required": True,
                        "style": "matrix",
                        "schema": {"type": "string"},
                    },
                )
            }
        },
        "get_item",
        {"id": "v1.2/a b"},
    )

    assert request.url.raw_path == b"/items/;id=v1%2E2%2Fa%20b"


@pytest.mark.parametrize("style", ["label", "matrix"])
async def test_styled_path_parameter_still_rejects_dot_segments(style: str) -> None:
    """Styled values keep the traversal guard applied to simple parameters."""
    with pytest.raises(Exception, match="dot segments"):
        await _capture_request(
            {
                "/items/{id}": {
                    "get": _operation(
                        "get_item",
                        {
                            "name": "id",
                            "in": "path",
                            "required": True,
                            "style": style,
                            "schema": {"type": "string"},
                        },
                    )
                }
            },
            "get_item",
            {"id": ".."},
        )
