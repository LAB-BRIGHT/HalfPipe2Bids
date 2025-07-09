from __future__ import annotations

import shutil
import json
import pandas as pd
import argparse

from pathlib import Path
from typing import Sequence
from nilearn.plotting import find_parcellation_cut_coords


from halfpipe2bids import __version__
from halfpipe2bids import utils as hp2b_utils
from halfpipe2bids.logger import hp2b_logger
from nilearn.connectivity import ConnectivityMeasure

hp2b_log = hp2b_logger()


def set_verbosity(verbosity: int | list[int]) -> None:
    if isinstance(verbosity, list):
        verbosity = verbosity[0]
    if verbosity == 0:
        hp2b_log.setLevel("ERROR")
    elif verbosity == 1:
        hp2b_log.setLevel("WARNING")
    elif verbosity == 2:
        hp2b_log.setLevel("INFO")
    elif verbosity == 3:
        hp2b_log.setLevel("DEBUG")


def global_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Convert neuroimaging data from the HalfPipe format to the "
            "standardized BIDS (Brain Imaging Data Structure) format."
        ),
    )
    parser.add_argument(
        "halfpipe_dir",
        action="store",
        type=Path,
        help="The directory with the HALFPipe output.",
    )
    parser.add_argument(
        "output_dir",
        action="store",
        type=Path,
        help="The directory where the output files should be stored.",
    )
    parser.add_argument(
        "analysis_level",
        help="Level of the analysis that will be performed. Only group"
        " level is available.",
        choices=["group"],
    )
    parser.add_argument(
        "--denoise-metadata",
        help="Add extra metadata about denoising info.",
        action="store_true",
    )
    parser.add_argument(
        "--impute-nan",
        help="Imputation and bad ROI removal.",
        action="store_true",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=__version__,
    )
    parser.add_argument(
        "--verbosity",
        help="Verbosity level.",
        required=False,
        choices=[0, 1, 2, 3],
        default=2,
        type=int,
        nargs=1,
    )
    return parser


def workflow(args: argparse.Namespace) -> None:
    hp2b_log.info(vars(args))
    output_dir = args.output_dir
    halfpipe_dir = args.halfpipe_dir

    path_derivatives = halfpipe_dir / "derivatives"
    path_halfpipe_timeseries = path_derivatives / "halfpipe"
    path_fmriprep = path_derivatives / "fmriprep"
    path_atlas_label = (
        halfpipe_dir / "atlas" / "atlas-Schaefer2018Combined_dseg.tsv"
    )
    path_atlas_nii = (
        halfpipe_dir / "atlas" / "atlas-Schaefer2018Combined_dseg.nii.gz"
    )
    path_halfpipe_spec = halfpipe_dir / "spec.json"

    set_verbosity(args.verbosity)

    with open(path_halfpipe_spec, "r") as f:
        halfpipe_spec = json.load(f)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    hp2b_log.info("Create dataset-level metadata.")
    hp2b_utils.create_dataset_metadata_json(
        output_dir, halfpipe_spec, path_atlas_nii
    )
    all_files = path_halfpipe_timeseries.glob("sub-*/**/*.*")

    # copy all files to the output directory
    for src in all_files:
        dst = hp2b_utils.get_bids_filename(src, output_dir)
        if not dst.parent.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
        hp2b_log.info(f"Renaming {src} to {dst}")
        if ".tsv" == src.suffix:
            mat = pd.read_csv(src, sep="\t", header=None, na_values="nan")
            mat.columns += 1  # add columns and use atlas index
            mat.to_csv(dst, index=False, sep="\t", na_rep="nan")
        else:
            shutil.copy2(src, dst)  # copy2 to preserve metadata

    if args.denoise_metadata:
        # populate timeseries.json with extra information
        for ts_jsons in output_dir.glob("sub-*/**/*_timeseries.json"):
            hp2b_utils.populate_timeseries_json(
                ts_jsons, path_fmriprep, halfpipe_spec
            )
        seg_meta_json = list(output_dir.glob("seg-*.json"))[0]
        coords = find_parcellation_cut_coords(path_atlas_nii)
        atlas_label = pd.read_csv(
            path_atlas_label, sep="\t", header=None, index_col=0
        )
        atlas_label.columns = ["parcel_name"]
        df_coords = pd.DataFrame(
            coords, columns=["x", "y", "z"], index=atlas_label.index
        )
        atlas_label = pd.concat([atlas_label, df_coords], axis=1)
        atlas_label["parcel_index"] = [
            i + 1 for i in range(atlas_label.shape[0])
        ]
        atlas_label.to_csv(
            output_dir / f"{seg_meta_json.stem}.tsv", index=False, sep="\t"
        )

    if args.impute_nan:
        parcel_removal_threshold = 0.5
        atlas_label = pd.read_csv(
            path_atlas_label, sep="\t", header=None, index_col=0
        ).index.tolist()
        timeseries_paths = list(output_dir.glob("sub-*/**/*_timeseries.tsv"))
        # find parcels coverage stats at dataset level
        dataset_nan_info = hp2b_utils.find_bad_rois(
            timeseries_paths, atlas_label
        )
        labels_to_drop = (
            dataset_nan_info[dataset_nan_info > parcel_removal_threshold]
            .dropna()
            .index.tolist()
        )
        labels_to_keep = [
            str(label)
            for label in atlas_label
            if str(label) not in labels_to_drop
        ]

        # replace nan with row means (mean value of all parcels per TR)
        relmat_calculation = {
            "covariance": ConnectivityMeasure(kind="covariance"),
            "PearsonCorrelation": ConnectivityMeasure(kind="correlation"),
        }
        for p in timeseries_paths:
            df = pd.read_csv(p, sep="\t", header=0, na_values="nan")
            row_means = df.mean(axis=1, skipna=True)
            df_imputed = df.T.fillna(row_means).T
            df_imputed.loc[:, labels_to_keep].to_csv(
                p, index=False, sep="\t", na_rep="nan"
            )
            # recreate the functional connectivity
            for relmat_type in relmat_calculation:
                dst = Path(
                    str(p).replace("timeseries", f"meas-{relmat_type}_relmat")
                )
                relmat = relmat_calculation[relmat_type].fit_transform(
                    df_imputed.values
                )
                df_relmat = pd.DataFrame(relmat, columns=df_imputed.columns)
                df_relmat.to_csv(dst, index=False, sep="\t", na_rep="nan")

        seg_meta_json = list(output_dir.glob("seg-*.json"))[0]
        seg_meta_tsv = list(output_dir.glob("seg-*.tsv"))[0]
        seg_meta_df = pd.read_csv(
            seg_meta_tsv, sep="\t", header=0, index_col="parcel_index"
        )

        with open(seg_meta_json, "r") as f:
            seg_metadata = json.load(f)
        seg_metadata_exta = {
            "ParcelExclusionThreashold": parcel_removal_threshold,
            "ParcelsRemoved": labels_to_drop,
        }
        seg_metadata.update(seg_metadata_exta)
        with open(seg_meta_json, "w") as f:
            json.dump(seg_metadata, f, indent=4)
        dataset_nan_info.index = seg_meta_df.index
        seg_meta_df = pd.concat([seg_meta_df, dataset_nan_info], axis=1)
        seg_meta_df.to_csv(seg_meta_tsv, sep="\t")


def main(argv: None | Sequence[str] = None) -> None:
    """Entry point."""
    parser = global_parser()
    args = parser.parse_args(argv)
    workflow(args)
