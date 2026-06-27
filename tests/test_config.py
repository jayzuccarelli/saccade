"""Config kills the launcher script: it auto-loads a .env and assembles the RTSP
URL from parts so creds (and a password full of url-breaking symbols) never have
to live in a hand-run shell file."""

from saccade.config import Config, _apply_dotenv


def test_apply_dotenv_parses_and_unquotes(tmp_path, monkeypatch):
    monkeypatch.delenv("SACCADE_FOO", raising=False)
    env = tmp_path / ".env"
    env.write_text('# a comment\n\nSACCADE_FOO=bar\nexport SACCADE_QUOTED="baz qux"\n')
    parsed = _apply_dotenv(str(env))
    assert parsed["SACCADE_FOO"] == "bar"
    assert parsed["SACCADE_QUOTED"] == "baz qux"  # quotes stripped, export honored
    import os

    assert os.environ["SACCADE_FOO"] == "bar"


def test_apply_dotenv_does_not_clobber_real_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SACCADE_FOO", "from-real-env")
    env = tmp_path / ".env"
    env.write_text("SACCADE_FOO=from-file\n")
    _apply_dotenv(str(env))
    import os

    assert os.environ["SACCADE_FOO"] == "from-real-env"  # real env always wins


def test_rtsp_url_assembled_from_parts_with_encoding():
    c = Config(
        rtsp_url="",
        rtsp_user="admin",
        rtsp_password="p@ss/w:rd",
        rtsp_host="10.0.0.9:554",
        rtsp_path="/h264Preview_01_sub",
    )
    # the symbols are percent-encoded so they can't break the URL
    assert c.rtsp_url == "rtsp://admin:p%40ss%2Fw%3Ard@10.0.0.9:554/h264Preview_01_sub"


def test_explicit_rtsp_url_is_not_overridden():
    c = Config(rtsp_url="rtsp://given/stream", rtsp_host="10.0.0.9", rtsp_password="x")
    assert c.rtsp_url == "rtsp://given/stream"  # an explicit URL wins over the parts
