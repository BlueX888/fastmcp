"""Regression tests: logging via ctx.info()/ctx.log() inside a background task.

A SEP-2663 worker Context is built without a live session (input and status
are polled, so the worker needs no back-channel). Logging must degrade the
way the other session-scoped Context APIs already do — gate the record by
the submitting session's snapshotted session id, mirror it to the
``fastmcp.server.context.to_client`` logger, and skip only the client
notification — rather than raising ``RuntimeError`` and aborting the tool
body halfway through.
"""

from __future__ import annotations

import logging

from fastmcp import FastMCP
from fastmcp.server.context import Context, to_client_logger
from fastmcp.server.dependencies import _background_task_session_id
from fastmcp_tasks import TasksExtension
from tests.tasks.task_helpers import (
    running_task_server,
    submit_task,
    wait_for_task,
)


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

    async def test_log_degrades_and_gates_on_snapshot_session_id(self):
        """A worker Context has no live session: log() must not raise, the
        logging/setLevel gate keyed by the snapshotted session id must still
        apply, and surviving records must mirror to the to_client logger."""
        mcp = FastMCP("test")
        ctx = Context(mcp, task_id="test-task-log")
        assert ctx.is_background_task is True
        assert ctx.request_context is None

        session_id = "session-abc"
        _background_task_session_id.set(session_id)
        mcp._client_log_levels[session_id] = "error"

        mirrored: list[str] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                mirrored.append(record.getMessage())

        handler = _Capture(level=logging.DEBUG)
        to_client_logger.addHandler(handler)
        try:
            await ctx.info("info msg")  # below the gate: suppressed
            await ctx.error("error msg")  # at the gate: mirrored server-side
        finally:
            to_client_logger.removeHandler(handler)

        assert [m for m in mirrored if "info msg" in m] == []
        assert [m for m in mirrored if "error msg" in m] == [
            "Sending ERROR to client: error msg"
        ]
