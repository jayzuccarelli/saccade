"""Speak through Home Assistant: synthesize the line, serve the clip on the LAN,
and tell a media_player to play it.

Decoupled from HA's filesystem — saccade serves its own audio over a small HTTP
server and HA just fetches the URL, so there's no dependency on HA's `www` dir or
host. Built and tested against a Sonos ("Den"). Any `media_player` works.

`tts` is anything with `async synthesize(text) -> Path` (e.g. GeminiTTSSpeaker).
The HTTP server starts lazily on the first utterance and serves the clip dir.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


class _QuietHandler(SimpleHTTPRequestHandler):
    """A media player closes the socket the instant it has the clip, which surfaces
    mid-send as BrokenPipe/ConnectionReset. That's expected, not an error — swallow
    it (and the per-fetch access log) so a normal playback doesn't dump a traceback
    every time saccade speaks."""

    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def log_message(self, *args: Any) -> None:
        pass


def _lan_ip() -> str:
    """The address other devices on the LAN can reach this box at.

    Falls back to loopback when there's no egress route (e.g. a sandbox) — with
    no LAN there's nothing for HA to fetch from anyway, so loopback is the only
    sane default and beats crashing the constructor."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # no packets sent; just picks the egress iface
        addr: str = s.getsockname()[0]
        return addr
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class HomeAssistantSpeaker:
    def __init__(
        self,
        tts: Any,  # anything with `async synthesize(text) -> Path` (e.g. GeminiTTSSpeaker)
        ha_url: str,
        token: str,
        entity_id: str,
        serve_host: str = "",
        serve_port: int = 8189,
    ) -> None:
        self.tts = tts
        self.ha_url = ha_url.rstrip("/")
        self.token = token
        self.entity_id = entity_id
        self.serve_host = serve_host or _lan_ip()
        self.serve_port = serve_port
        self._server: ThreadingHTTPServer | None = None

    def _ensure_server(self, directory: Path) -> None:
        if self._server is not None:
            return
        handler = partial(_QuietHandler, directory=str(directory))
        self._server = ThreadingHTTPServer(("0.0.0.0", self.serve_port), handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def _play_media(self, url: str) -> None:
        body = json.dumps(
            {
                "entity_id": self.entity_id,
                "media_content_id": url,
                "media_content_type": "music",
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.ha_url}/api/services/media_player/play_media",
            data=body,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10).read()

    async def say(self, text: str) -> None:
        path = await self.tts.synthesize(text)
        self._ensure_server(path.parent)
        assert self._server is not None  # _ensure_server just set it
        port = self._server.server_address[1]  # actual bound port (serve_port=0 → ephemeral)
        url = f"http://{self.serve_host}:{port}/{path.name}"
        await asyncio.to_thread(self._play_media, url)
        print(f"\n    \033[1m\033[96m💬  {text}\033[0m   🔊 {self.entity_id}\n")
