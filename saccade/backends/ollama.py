"""Ollama backend: local, private, free. Stdlib only.

Talks to a local Ollama instance (http://localhost:11434 by default; override
with OLLAMA_HOST or SACCADE_OLLAMA_HOST). No new dependencies: urllib + json.

    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull gemma3:4b           # ~3GB, multimodal, fast (Glance default)
    ollama pull gemma3:12b          # smarter (Focus default)

    SACCADE_GLANCE_BACKEND=ollama SACCADE_FOCUS_BACKEND=ollama python -m saccade

Structured output uses Ollama's native `format` field (a JSON Schema), enforced
by the runtime, not prompt-engineered. Vision frames go as base64 in `images`.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import subprocess
import threading
import time
from typing import Any
from urllib import error, request

from saccade.schema import Frame, JsonSchema, heard_text

_DEFAULT_HOST = "http://localhost:11434"
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]")

# One attempt per process. A daemon that won't stay up shouldn't be respawned on
# every glance, and the second failure is the one worth reporting.
#
# Locked, because the two tiers run concurrently and `_post` goes through
# asyncio.to_thread: review caught that Glance and Focus failing on the same tick
# could both pass an unguarded check and spawn a daemon each. The lock covers the
# whole attempt, not just the flag, so the second thread waits for the first
# server to come up instead of racing it with a second one.
_start_lock = threading.Lock()
_start_result: bool | None = None


class OllamaError(RuntimeError):
    """Ollama refused, with the command that fixes it. The loop prints the
    message verbatim, so it has to read like an instruction: a bare
    `URLError: [Errno 61] Connection refused` on every tick tells you nothing."""


def _is_local(host: str) -> bool:
    return host.split("://", 1)[-1].split("/")[0].rsplit(":", 1)[0] in _LOCAL_HOSTS


def _start_daemon(host: str) -> bool:
    """Start a local Ollama that isn't running, once, and say whether it came up.

    Same argument as the setup wizard makes: we know the command, we can run it,
    and printing `ollama serve` at someone on every tick is not a fix. Scoped hard,
    because this runs unattended where setup doesn't: a local host only, so there's
    no remote server to duplicate; only when nothing is answering, so there's
    nothing to collide with; and once per process, so a daemon that dies on startup
    doesn't get a fresh one every second.

    Announced, not silent. Starting a background process someone didn't ask for is
    exactly the kind of thing that has to show up in the log."""
    global _start_result
    if not _is_local(host) or not shutil.which("ollama"):
        return False
    with _start_lock:
        if _start_result is not None:
            return _start_result
        _start_result = False
        print("[ollama] not running; starting it")
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            return False
        for _ in range(20):
            time.sleep(0.25)
            try:
                with request.urlopen(f"{host}/api/tags", timeout=0.5):
                    print("[ollama] up")
                    _start_result = True
                    return True
            except OSError:  # URLError subclasses OSError
                continue
        return False


class OllamaBackend:
    def __init__(self, model: str, host: str | None = None, timeout: float = 120.0):
        self.model = model
        self.host = (
            host
            or os.environ.get("SACCADE_OLLAMA_HOST")
            or os.environ.get("OLLAMA_HOST")
            or _DEFAULT_HOST
        ).rstrip("/")
        self.timeout = timeout

    async def complete(
        self, prompt: str, frames: list[Frame], schema: JsonSchema | None = None
    ) -> str:
        message: dict[str, Any] = {"role": "user", "content": prompt + heard_text(frames)}
        images = [base64.b64encode(f.image).decode() for f in frames if f.image]
        if images:
            message["images"] = images

        body: dict[str, Any] = {"model": self.model, "messages": [message], "stream": False}
        if schema:
            body["format"] = schema

        return await asyncio.to_thread(self._post, body)

    def _post(self, body: dict[str, Any]) -> str:
        req = request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except error.HTTPError as e:  # subclasses URLError; catch it first
            if e.code == 404:
                raise OllamaError(
                    f"Ollama has no model {self.model!r}; pull it: ollama pull {self.model}"
                ) from e
            raise OllamaError(f"Ollama returned HTTP {e.code}") from e
        except error.URLError as e:
            if _start_daemon(self.host):
                return self._post(body)  # the flag makes this recurse at most once
            raise OllamaError(
                f"Ollama isn't reachable at {self.host}; start it: ollama serve"
            ) from e
        content: str = data.get("message", {}).get("content", "")
        return content
