"""The default Speaker: print the utterance. No audio, no key, no deps."""

from __future__ import annotations


class PrintSpeaker:
    async def say(self, text: str) -> None:
        print(f"\n    \033[1m\033[96m💬  {text}\033[0m\n")
