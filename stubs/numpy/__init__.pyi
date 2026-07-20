# Shadows numpy's real stubs, which are written in 3.12 syntax (PEP 695 `type`
# statements) that mypy can't parse while saccade targets 3.10. It's a parse
# error, so it aborts the whole run rather than one file, and it lands on anyone
# who installs the audio extra or piper-tts. We use numpy for exactly one thing
# (framing PCM for sounddevice), so treating it as Any costs nothing and keeps
# `make check` green without dropping the 3.10 target for everything else.
from typing import Any

def __getattr__(name: str) -> Any: ...
