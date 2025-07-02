"""
Simple code to smoke test the functionality.
"""

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
    assert (
        "Convert neuroimaging data from the HalfPipe format to" in captured.out
    )


@pytest.mark.smoke
def test_smoke(tmp_path, caplog):
    halfpipe_dir = (
        resources.files("halfpipe2bids")
        / "tests/data/dataset-ds000030_halfpipe1.2.3dev"
    )
    output_dir = tmp_path / "output"

    main(
        [
            str(halfpipe_dir),
            str(output_dir),
            "group",
        ]
    )
    output_folder = output_dir / "sub-10159/func"
    base = "sub-10159_task-rest_seg-schaefer400417"
    ts_base = base + "_desc-denoisecorrMatrix1"
    relmat_file = output_folder / (
        ts_base + "_meas-PearsonCorrelation_relmat.tsv"
    )
    # checking if relmat file exists
    if not relmat_file.exists():
        raise FileNotFoundError(
            f"Expected file not found: {relmat_file}\nAvailable files:\n"
            + "\n".join(str(p) for p in output_folder.glob("*"))
        )

    relmat = pd.read_csv(relmat_file, sep="\t")
    # This is the number of ROI (columns) I got from the supposedly original file
    # TODO: when the --impute-nans option is added, this test should pass
    assert relmat.shape[1] == 434
    json_file = output_folder / (ts_base + "_timeseries.json")
    assert json_file.exists()
    with open(json_file, "r") as f:
        content = json.load(f)
        # the unit is Hz, for TR= 2s, the sampling frequency is 0.5 Hz
        assert content.get("SamplingFrequency") == 0.5

    # TODO: when the --impute-nans option is added, create a test for
    # the relmat with NaNs replaced by grand mean
