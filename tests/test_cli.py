"""CLI smoke tests."""

from __future__ import annotations

import pytest

from gaussforge.cli import main


def test_cli_info(capsys):
    code = main(["info"])
    assert code == 0
    out = capsys.readouterr().out
    assert "GaussForge" in out


def test_cli_rejects_unknown_command():
    with pytest.raises(SystemExit):
        main(["bogus"])
