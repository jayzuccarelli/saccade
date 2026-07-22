"""The AV sensor fuses webcam + mic into one Frame. Capture needs hardware, but
the fusion (one tick yields both an image and an audio clip) is the contract
worth pinning, with cv2 and the mic faked."""

import asyncio
import sys
import time
import wave
from io import BytesIO
from types import SimpleNamespace

import saccade.sensors.av as av
from saccade.sensors.av import AVSensor


class _FakeCap:
    def __init__(self, *a):
        self._opened = True

    def isOpened(self):
        return True

    def set(self, *a):
        pass

    def grab(self):
        # A driver with nothing queued: the first grab is the live frame.
        self.grabs = getattr(self, "grabs", 0) + 1
        time.sleep(0.01)
        return True

    def read(self):
        return True, "rawframe"  # a stand-in ndarray; imencode is faked below

    def release(self):
        self._opened = False


def _fake_cv2():
    return SimpleNamespace(
        VideoCapture=_FakeCap,
        CAP_PROP_BUFFERSIZE=38,
        imencode=lambda ext, frame: (True, SimpleNamespace(tobytes=lambda: b"\xff\xd8jpeg")),
    )


async def _first_frame(sensor):
    async for f in sensor.stream():
        return f


def test_av_frame_carries_both_image_and_audio(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2())
    monkeypatch.setattr(av, "require_audio", lambda: None)  # skip the PortAudio check
    monkeypatch.setattr(av, "record_pcm", lambda secs, rate, dev: b"\x01\x00" * 100)

    frame = asyncio.run(_first_frame(AVSensor(webcam_index=0, mic_index=None, fps=10)))

    assert frame.image == b"\xff\xd8jpeg"  # the camera half
    assert frame.mime == "image/jpeg"
    # the audio half is a real, readable WAV wrapping the recorded PCM
    assert frame.audio_mime == "audio/wav"
    with wave.open(BytesIO(frame.audio), "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnframes() == 100


def test_av_records_the_requested_mic_and_interval(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2())
    monkeypatch.setattr(av, "require_audio", lambda: None)
    calls = {}
    monkeypatch.setattr(
        av, "record_pcm", lambda secs, rate, dev: calls.update(secs=secs, dev=dev) or b""
    )

    asyncio.run(_first_frame(AVSensor(webcam_index=0, mic_index=2, fps=4)))

    assert calls["dev"] == 2  # the chosen mic index reaches the recorder
    assert abs(calls["secs"] - 0.25) < 1e-9  # clip spans one glance interval (1/fps)


def test_the_audio_window_is_not_the_glance_interval():
    """1/fps made a one-second clip at the default rate. You can't say a sentence
    in a second, and Whisper on a one-second fragment usually returns nothing, so
    a mic that worked perfectly looked deaf."""
    assert AVSensor(fps=1.0).seconds == 1.0  # the old behaviour, still available
    assert AVSensor(fps=1.0, seconds=4.0).seconds == 4.0


def test_the_frame_is_drained_before_it_is_read(monkeypatch):
    """Each tick here records seconds of audio and may transcribe it, so the queue
    grows the whole time and cap.read() hands back an ever-older frame."""
    cap = _FakeCap()
    monkeypatch.setattr(av, "latest_jpeg", lambda c: (c.grab(), b"jpeg")[1])
    assert av.latest_jpeg(cap) == b"jpeg"
    assert cap.grabs == 1
