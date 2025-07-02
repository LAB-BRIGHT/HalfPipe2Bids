import os
import json
import pandas as pd
import logging
from nilearn.signal import clean
from nilearn import plotting
import re
from halfpipe2bids import __version__

hp2b_log = logging.getLogger("halfpipe2bids")
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


def load_label_schaefer(path_label_schaefer):
    # TODO: documentation and eventually remove - we what this to work with
    # different atlases
    return list(pd.read_csv(path_label_schaefer, sep="\t", header=None)[1])


def get_strategy_confounds(spec_path):
    # TODO: documentation
    with open(spec_path, "r") as f:
        data = json.load(f)

    setting_to_confounds = {
        s["name"]: s.get("confounds_removal", [])
        for s in data.get("settings", [])
    }

    strategy_confounds = {}
    for feature in data.get("features", []):
        strategy_name = feature.get("name")
        setting_name = feature.get("setting")
        strategy_confounds[strategy_name] = setting_to_confounds.get(
            setting_name, []
        )

    return strategy_confounds


def impute_and_clean(df):
    # TODO: documentation and what's the imputation method?
    row_means = df.mean(axis=1, skipna=True)
    df_filled = df.T.fillna(row_means).T

    if df_filled.isna().any().any():
        hp2b_log.warning("Certaines valeurs n'ont pas pu être imputées.")

    cleaned = clean(
        df_filled.values, detrend=True, standardize="zscore_sample"
    )
    return pd.DataFrame(cleaned, columns=df.columns, index=df.index)


def remove_bad_rois(dict_timeseries, label_schaefer, threshold=0.5):
    # TODO: documentation
    nan_counts = {label: 0 for label in label_schaefer}
    total_subjects = len(dict_timeseries)

    for df in dict_timeseries.values():
        for label in label_schaefer:
            if label in df.columns and df[label].isna().all():
                nan_counts[label] += 1

    df_nan_prop = pd.DataFrame(
        {
            "ROI": list(nan_counts.keys()),
            "proportion_nan": [
                nan_counts[label] / total_subjects for label in label_schaefer
            ],
        }
    )

    labels_to_drop = df_nan_prop[df_nan_prop["proportion_nan"] > threshold][
        "ROI"
    ].tolist()

    for key in dict_timeseries:
        dict_timeseries[key] = dict_timeseries[key].drop(
            columns=labels_to_drop, errors="ignore"
        )

    return labels_to_drop


def get_coords(volume_path, label_schaefer, labels_to_drop):
    # TODO: documentation
    coords = plotting.find_parcellation_cut_coords(volume_path)
    df_coords = pd.DataFrame(
        coords, index=label_schaefer, columns=["x", "y", "z"]
    )
    return df_coords[~df_coords.index.isin(labels_to_drop)]


def create_dataset_metadata_json(output_dir) -> None:
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

    file_output_dir = output_dir / f"sub-{entities['sub']}" / "func"
    if extension == ".gz":
        return file_output_dir / src.name

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
    return file_output_dir / f"{new_basename}{new_suffix_info}"
