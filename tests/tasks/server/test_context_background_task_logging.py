"""Regression tests: logging via ctx.info()/ctx.log() inside a background task.

A SEP-2663 worker Context is built without a live session (input and status
are polled, so the worker needs no back-channel). Logging must degrade the
way the other session-scoped Context APIs already do — gate the record by
the submitting session's ``logging/setLevel`` key, mirror it to the
``fastmcp.server.context.to_client`` logger, and skip only the client
notification — rather than raising ``RuntimeError`` and aborting the tool
body halfway through.

The gate key is not ``context.session_id``: setLevel stores it under
``_log_level_session_key(session)`` (``connection.session_id``, else the
``"__no_session__"`` sentinel), and the snapshot carries that exact key so a
worker's generated UUID or cached state prefix cannot miss it.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest
from fastmcp_tasks.context import TaskContextSnapshot
from mcp import ServerSession

import fastmcp.server.dependencies as server_dependencies
from fastmcp import FastMCP
from fastmcp.server.context import Context, to_client_logger
from fastmcp.server.dependencies import _background_task_session_id
from fastmcp_tasks import TasksExtension
from tests.tasks.task_helpers import (
    running_task_server,
    submit_task,
    wait_for_task,
)


def _log_level_key_var():
    """The key-alignment ContextVar, or ``None`` on a pre-fix checkout.

    Resolved dynamically so this module still collects on a checkout that
    predates the symbol: the missing alignment must fail a test's assertion,
    not the import, so the red receipt reflects behavior.
    """
    return getattr(server_dependencies, "_background_task_log_level_session_key", None)


async def _log(coro: Awaitable[None]) -> None:
    """Await a Context.log call, failing the test if the old bug re-raises."""
    try:
        await coro
    except RuntimeError as exc:
        raise AssertionError(
            f"ctx logging raised RuntimeError in a background task: {exc}"
        ) from exc


def _capture_mirror() -> tuple[list[str], logging.Handler]:
    """Collect the messages mirrored to the to_client logger."""

    class _Capture(logging.Handler):
        def __init__(self) -> None:
            super().__init__(level=logging.DEBUG)
            self.messages: list[str] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.messages.append(record.getMessage())

    handler = _Capture()
    to_client_logger.addHandler(handler)
    return handler.messages, handler


class TestLoggingInBackgroundTask:
    """End-to-end logging in a background task, via the shared helpers."""

    async def test_logging_in_background_task(self):
        """ctx.info()/ctx.debug() in a worker must not abort the tool body."""
        mcp = FastMCP("log-test")
        mcp.add_extension(TasksExtension())

        @mcp.tool(task=True)
        async def logging_task(message: str, ctx: Context) -> str:
            await ctx.info(f"processing {message}")
            await ctx.debug("still alive")
            return f"ok:{message}"

        async with running_task_server(mcp):
            created = await submit_task(mcp, "logging_task", {"message": "x"})
            final = await wait_for_task(mcp, created.task_id)

        assert final.status == "completed"
        assert final.result is not None
        assert final.result["isError"] is False, final.result["content"][0]["text"]
        assert final.result["structuredContent"] == {"result": "ok:x"}


class TestSessionlessWorkerContextLogging:
    """Unit coverage for the session-less (worker) Context log path."""

    async def test_log_degrades_and_gates_on_captured_setlevel_key(self):
        """A worker Context has no live session: log() must not raise, the
        logging/setLevel gate keyed by the snapshot's exact key must still
        apply, and surviving records must mirror to the to_client logger."""
        mcp = FastMCP("test")
        ctx = Context(mcp, task_id="test-task-log")
        assert ctx.is_background_task is True
        assert ctx.request_context is None

        session_id = "session-abc"
        # The id ctx.session_id would snapshot is not the setLevel key; the
        # snapshot separately carries the exact key setLevel wrote.
        level_key_var = _log_level_key_var()
        assert level_key_var is not None, (
            "fastmcp.server.dependencies is missing the background-task "
            "setLevel key ContextVar"
        )
        session_token = _background_task_session_id.set("generated-uuid")
        key_token = level_key_var.set(session_id)
        mcp._client_log_levels[session_id] = "error"

        mirrored, handler = _capture_mirror()
        try:
            await _log(ctx.info("info msg"))  # below the gate: suppressed
            await _log(ctx.error("error msg"))  # at the gate: mirrored server-side
        finally:
            to_client_logger.removeHandler(handler)
            level_key_var.reset(key_token)
            _background_task_session_id.reset(session_token)

        assert [m for m in mirrored if "info msg" in m] == []
        assert [m for m in mirrored if "error msg" in m] == [
            "Logging ERROR server-side: error msg"
        ]

    async def test_setlevel_gate_survives_a_diverging_snapshot_session_id(self):
        """The fallback must resolve the setLevel sentinel, not the snapshot id.

        STDIO/SSE/in-memory setLevel writes ``"__no_session__"`` while
        ``ctx.session_id`` would snapshot a generated UUID. A worker with no
        exact key captured (e.g. a legacy snapshot) must still honor the
        client's minimum instead of silently logging below it.
        """
        mcp = FastMCP("test")
        ctx = Context(mcp, task_id="test-task-keyalign")
        assert ctx.is_background_task is True

        # What logging/setLevel stored for a session-less connection.
        mcp._client_log_levels["__no_session__"] = "error"
        # What the task snapshot's session_id would be: a generated UUID.
        session_token = _background_task_session_id.set(str(uuid4()))

        mirrored, handler = _capture_mirror()
        try:
            await _log(ctx.info("info msg"))  # below the gate: suppressed
            await _log(ctx.error("error msg"))  # at the gate: mirrored server-side
        finally:
            to_client_logger.removeHandler(handler)
            _background_task_session_id.reset(session_token)

        assert [m for m in mirrored if "info msg" in m] == []
        error_records = [m for m in mirrored if "error msg" in m]
        assert error_records == ["Logging ERROR server-side: error msg"]
        # There is no live session, so the mirrored record must not claim a
        # client send.
        assert all("to client" not in m for m in error_records)

    @pytest.mark.parametrize(
        ("connection_session_id", "expected_key"),
        [(None, "__no_session__"), ("http-session-1", "http-session-1")],
    )
    async def test_snapshot_capture_records_the_setlevel_key(
        self, connection_session_id: str | None, expected_key: str
    ):
        """Task submission captures the live session's setLevel key verbatim."""
        mcp = FastMCP("test")
        connection = SimpleNamespace(session_id=connection_session_id, state={})
        session = SimpleNamespace(_connection=connection)
        ctx = Context(mcp, session=cast(ServerSession, session))

        async with ctx:
            snapshot = TaskContextSnapshot.capture()

        assert getattr(snapshot, "log_level_session_key", None) == expected_key, (
            "the task snapshot did not capture the logging/setLevel gate key"
        )
