import io
import sys

import pip._internal.cli.main as pip_cli

import app.main as application


def test_packaged_pip_command_restores_missing_windowed_stdio(monkeypatch):
    opened_streams = {}

    def fake_fdopen(file_descriptor, mode, **_options):
        stream = io.StringIO()
        opened_streams[mode, file_descriptor] = stream
        return stream

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(
        sys, "argv", ["DORA.exe", "--dora-editor-pip", "--version"]
    )
    monkeypatch.setattr(sys, "stdin", None)
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    monkeypatch.setattr(application.os, "fstat", lambda _fd: object())
    monkeypatch.setattr(application.os, "dup", lambda fd: fd + 10)
    monkeypatch.setattr(application.os, "fdopen", fake_fdopen)

    def fake_pip_main(arguments):
        sys.stdout.write("pip output is available")
        sys.stderr.write("pip errors are available")
        assert arguments == ["--version"]
        return 0

    monkeypatch.setattr(pip_cli, "main", fake_pip_main)

    result = application._run_packaged_editor_command()

    assert result == 0
    assert opened_streams[("w", 11)].getvalue() == "pip output is available"
    assert opened_streams[("w", 12)].getvalue() == "pip errors are available"
    assert sys.argv == ["DORA.exe", "--dora-editor-pip", "--version"]
