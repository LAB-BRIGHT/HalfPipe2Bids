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
    cmd = [
        str(halfpipe_dir),
        str(output_dir),
        "group",
    ]
    main(cmd)
    output_folder = output_dir / "sub-10159/func"
    base = "sub-10159_task-rest_seg-schaefer400"
    ts_base = base + "_desc-corrMatrix1"
    relmat_file = output_folder / (
        ts_base + "_meas-PearsonCorrelation_relmat.tsv"
    )
    # checking if relmat file exists
    assert relmat_file.exists()
    relmat = pd.read_csv(relmat_file, sep="\t")
    # This is the number of ROI (columns) I got from the supposedly original file
    assert relmat.shape[1] == 434
    json_file = output_folder / (ts_base + "_timeseries.json")
    assert json_file.exists()
    with open(json_file, "r") as f:
        content = json.load(f)
        # the unit is Hz, for TR= 2s, the sampling frequency is 0.5 Hz
        # however, when no flags are passed, this is just copying the original
        # file, hence ,mistake remains.
        assert content.get("SamplingFrequency") == 2

    # TODO: when the --impute-nans option is added, create a test for
    # the relmat with NaNs replaced by grand mean

    main(cmd + ["--denoise-meta"])
    assert json_file.exists()
    with open(json_file, "r") as f:
        content = json.load(f)
        # the unit is Hz, for TR= 2s, the sampling frequency is 0.5 Hz
        assert content.get("SamplingFrequency") == 0.5
    relmat = pd.read_csv(relmat_file, sep="\t")
    # This is the number of ROI (columns) I got from the supposedly original file
    assert relmat.shape[1] == 434  # the content of the file untouched

    main(cmd + ["--impute-nan"])
    assert json_file.exists()
    with open(json_file, "r") as f:
        content = json.load(f)
        # the unit is Hz, for TR= 2s, the sampling frequency is 0.5 Hz
        # however, when no flags are passed, this is just copying the original
        # file, hence ,mistake remains.
        assert content.get("SamplingFrequency") == 2
    relmat = pd.read_csv(relmat_file, sep="\t")
    # This is the number of ROI (columns) I got from the supposedly original file
    assert relmat.shape[1] == 417  # ROI with too many subjects missing removed
