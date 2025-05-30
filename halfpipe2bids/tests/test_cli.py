"""
Simple code to smoke test the functionality.
"""

from pathlib import Path
from importlib import resources
import json
import pytest

import pandas as pd

from halfpipe2bids import __version__
from halfpipe2bids.main import main


def test_version(capsys):
    try:
        main(["-v"])
    except SystemExit:
        pass
    captured = capsys.readouterr()
    assert __version__ == captured.out.split()[0]


def test_help(capsys):
    try:
        main(["-h"])
    except SystemExit:
        pass
    captured = capsys.readouterr()
    assert "Convert neuroimaging data from the HalfPipe format to" in captured.out


@pytest.mark.smoke
def test_smoke(tmp_path, caplog):
    halfpipe_dir = resources.files("halfpipe2bids") / "tests/data/dataset-ds000030_halfpipe1.2.3dev"
    output_dir = tmp_path / "output"

    main(
        [
            str(halfpipe_dir),
            str(output_dir),
            "group",
        ]
    )
    # TODO: check outputs

