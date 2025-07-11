import os
import json
import pandas as pd
import re
from halfpipe2bids import __version__

from halfpipe2bids.logger import hp2b_logger

hp2b_log = hp2b_logger()
hp2b_url = "https://github.com/LAB-BRIGHT/HalfPipe2Bids"

suffix_converter = {"matrix": "relmat", "timeseries": "timeseries"}
measure_entity_converter = {
    "correlation": "PearsonCorrelation",
    "covariance": "covariance",
}
regex_bids_entity = r"([a-zA-Z]*)-([^_]*)"

dataset_description = {
    "BIDSVersion": "1.9.0",
    "License": None,
    "Name": None,
    "ReferencesAndLinks": [],
    "DatasetDOI": None,
    "DatasetType": "derivative",
    "GeneratedBy": [
        {
            "Name": "Halfpipe2Bids",
            "Version": __version__,
            "CodeURL": hp2b_url,
        }
    ],
    "HowToAcknowledge": f"Please refer to our repository: {hp2b_url}",
}

meas_meta = {
    "covariates": {
        "Measure": "Covariance",
        "MeasureDescription": "Covariance",
        "Weighted": False,
        "Directed": False,
        "ValidDiagonal": True,
        "StorageFormat": "Full",
        "NonNegative": "",
        "Code": "HALFPipe",
    },
    "PearsonCorrelation": {
        "Measure": "Pearson correlation",
        "MeasureDescription": "Pearson correlation",
        "Weighted": False,
        "Directed": False,
        "ValidDiagonal": True,
        "StorageFormat": "Full",
        "NonNegative": "",
        "Code": "HALFPipe",
    },
}


def get_subjects(path_halfpipe_timeseries):
    # TODO: documentation
    return [
        sub
        for sub in os.listdir(path_halfpipe_timeseries)
        if os.path.isdir(os.path.join(path_halfpipe_timeseries, sub))
    ]


def get_halfpipe_denoise_strategy_names(spec_path):
    # TODO: documentation
    with open(spec_path, "r") as f:
        data = json.load(f)

    strategy_names = []
    for feature in data.get("features", []):
        strategy_name = feature.get("name")
        strategy_names.append(strategy_name)
    return strategy_names


def regex_to_regressor(regex_confounds, confounds_columns):
    """
    Convert the list of regex patterns from HALFpipe to a list of regressors.
    Args:
        regex_confounds (list): List of regex patterns.
        confounds_columns (list): List of column names from confound file.
    Returns:
        list: List of confound columns based on fmriprep confound file.
    """
    # TODO: To be merged with get_strategy_confounds

    # Compile the regex pattern
    pattern = re.compile("|".join(regex_confounds))
    return [col for col in confounds_columns if pattern.fullmatch(col)]


def find_bad_rois(timeseries_paths, atlas_label, parcel_removal_threshold=0.5):
    """
    Find out how many subject miss the same roi report in proportion of the
    dataset.

    Args:
        timeseries_paths (list[Path]): Path to all time series data.
        atlas_label (List): Parcel index (starting from 1).
        parcel_removal_threshold (float): proportion of the dataset.
            1.0 = all subjects in the dataset miss a given parcel.
            0.0 = all subjects in the dataset has a given parcel.
            Default: 0.5

    Returns:
        pandas.DataFrame: proportion of the dataset with nan per parcel.
        List: labels to keep.
        List: labels to drop.
    """
    per_roi_nan_counter = {str(label): [0] for label in atlas_label}
    total_subjects = len(timeseries_paths)

    for p in timeseries_paths:
        df = pd.read_csv(p, sep="\t", header=0, index_col=0, na_values="nan")
        subject_roi_missing = (pd.isna(df).sum() / df.shape[0]) == 1
        for label in df.columns[subject_roi_missing]:
            per_roi_nan_counter[label][0] += 1

    df_nan_prop = pd.DataFrame(per_roi_nan_counter).T / total_subjects
    df_nan_prop.columns = ["proportion_missing_in_dataset"]
    labels_to_drop = (
        df_nan_prop[df_nan_prop > parcel_removal_threshold]
        .dropna()
        .index.tolist()
    )
    labels_to_keep = [
        str(label) for label in atlas_label if str(label) not in labels_to_drop
    ]
    return df_nan_prop, labels_to_keep, labels_to_drop


def create_dataset_metadata_json(
    output_dir, halfpipe_spec, path_atlas_nii
) -> None:
    """
    Create dataset-level metadata JSON files for BIDS.
    Args:
        output_dir (Path): path to the output directory where the JSON file
        will be saved.
    """
    # create the dataset_description.json file
    hp2b_log.info(f"Creating {output_dir / 'dataset_description.json'}")
    with open(output_dir / "dataset_description.json", "w") as f:
        json.dump(dataset_description, f, indent=4)

    for meas in meas_meta:
        meas_path = output_dir / f"meas-{meas}_relmat.json"
        with open(meas_path, "w") as f:
            json.dump(meas_meta[meas], f, indent=4)
        hp2b_log.info(f"Exported {meas} metadata to {meas_path}")

    seg_meta = {
        "File": entry
        for entry in halfpipe_spec["files"]
        if entry.get("suffix", False)
    }

    with open(
        output_dir / f"seg-{seg_meta['File']['tags']['desc']}.json", "w"
    ) as f:
        json.dump(seg_meta, f, indent=4)


def load_atlas_info_tsv(path_atlas_label):
    """
    Load original atlas parcel label and index tsv from halfpipe.
    The first column is the index, second the parcel label.
    There's no header in the file.

    Args:
        path_atlas_label (Path): Path to the file.

    Returns:
        pandas.DataFrame:
    """
    atlas_label = pd.read_csv(
        path_atlas_label, sep="\t", header=None, index_col=0
    )
    atlas_label.columns = ["parcel_name"]
    atlas_label.index.name = "parcel_index"
    return atlas_label


def get_bids_filename(src, output_dir):
    """
    Generates a BIDS-compliant filename based on the source file's name
    and the output directory.

    Args:
        src (Path): The source file path, which should contain BIDS
            entities in its name.
        output_dir (Path): The output directory where the BIDS file
            will be saved.

    Returns:
        Path: The BIDS-compliant file path.

    Raises:
        KeyError: If required BIDS entities (e.g., 'sub', 'task',
            'atlas', 'feature') are missing from the source filename.

    Notes:
        - The function extracts BIDS entities from the source filename
            using a regular expression.
        - It applies entity and suffix conversions according to BIDS
            conventions.
        - The output path is structured as:
            <output_dir>/sub-<subject>/func/<BIDS_filename>.
    """

    # rename files to match BIDS naming conventions
    entities = re.findall(regex_bids_entity, src.stem)
    entities = {entity[0]: entity[1] for entity in entities}
    extension = src.suffix
    suffix = src.stem.split("_")[-1]

    if entities.get("sub", False):
        output_dir = output_dir / f"sub-{entities['sub']}" / "func"

    if extension == ".gz":
        return output_dir / src.name

    if suffix in suffix_converter:
        suffix = suffix_converter[suffix]
    if "desc" in entities and entities["desc"] in measure_entity_converter:
        entities["desc"] = measure_entity_converter[entities["desc"]]

    # convert entities to a dictionary
    new_basename = (
        f"sub-{entities['sub']}_task-{entities['task']}_"
        f"seg-{entities['atlas']}_desc-{entities['feature']}_"
    )
    new_suffix_info = (
        f"meas-{entities['desc']}_{suffix}{extension}"
        if "desc" in entities
        else f"{suffix}{extension}"
    )
    return output_dir / f"{new_basename}{new_suffix_info}"


def populate_timeseries_json(path_timeseries_json, fmriprep_dir):
    """Add additional meta data for denoising metric calculation to the
    existing json file.

    Args:
        path_timeseries_json (Path): Path to the meta data file.
        fmriprep_dir (Path): Associated fmriprep directory.

    Returns:
        None
    """
    sub = path_timeseries_json.stem.split("sub-")[-1].split("_")[0]
    task = path_timeseries_json.stem.split("task-")[-1].split("_")[0]
    confound_file = (
        fmriprep_dir
        / f"sub-{sub}"
        / "func"
        / f"sub-{sub}_task-{task}_desc-confounds_timeseries.tsv"
    )
    confounds = pd.read_csv(confound_file, sep="\t")
    extra_meta = {}
    with open(path_timeseries_json, "r") as f:
        timeseries_meta = json.load(f)

    sampling_freq = timeseries_meta.get("SamplingFrequency", None)

    # convert sampling_freq from sec to Hz
    # TODO: this is an upstream issue that should be reported
    if sampling_freq is not None:
        sampling_freq = 1.0 / sampling_freq
    extra_meta["SamplingFrequency"] = sampling_freq

    # convert confound regressors
    denoise_setting = timeseries_meta["Setting"]

    extra_meta["ConfoundRegressors"] = regex_to_regressor(
        denoise_setting["ConfoundsRemoval"], confounds.columns.tolist()
    )
    extra_meta["NumberOfVolumesDiscardedByMotionScrubbing"] = len(
        regex_to_regressor(
            ["motion_outlier[0-9]+"], confounds.columns.tolist()
        )
    )
    extra_meta["MeanFramewiseDisplacement"] = confounds[
        "framewise_displacement"
    ].mean()
    extra_meta["MaxFramewiseDisplacement"] = (
        confounds["framewise_displacement"].max(),
    )
    timeseries_meta.update(extra_meta)
    with open(path_timeseries_json, "w") as f:
        json.dump(timeseries_meta, f, indent=4)
