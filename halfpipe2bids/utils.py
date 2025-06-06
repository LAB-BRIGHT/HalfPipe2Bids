import os
import json
import pandas as pd
import logging
from nilearn.signal import clean
from nilearn import plotting

hp2b_log = logging.getLogger("halfpipe2bids")
hp2b_url = "https://github.com/pbergeret12/HalfPipe2Bids/"


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


def crearte_dataset_metadata_json(output_dir) -> None:
    """
    Create dataset-level metadata JSON files for BIDS.
    Args:
        output_dir (Path): path to the output directory where the JSON file
        will be saved.
    """
    # export json file of common metadata for BIDS dataset
    summary_path = output_dir / "meas-PearsonCorrelation_relmat.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "Measure": "Pearson correlation",
                "MeasureDescription": "Pearson correlation",
                "Weighted": False,
                "Directed": False,
                "ValidDiagonal": True,
                "StorageFormat": "Full",
                "NonNegative": "",
                "Code": hp2b_url,
            },
            f,
            indent=4,
        )

    hp2b_log.info(f"Export terminé dans : {output_dir}")

    # Export du json de description de dataset

    json_dataset_description = {
        "BIDSVersion": "1.9.0",
        "License": None,
        "Name": None,
        "ReferencesAndLinks": [],
        "DatasetDOI": None,
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "Halfpipe2Bids",
                "Version": "0.1",
                "CodeURL": hp2b_url,
            }
        ],
        "HowToAcknowledge": f"Please refer to our repository: {hp2b_url}",
    }

    output_filename = "dataset_description.json"
    output_file = output_dir / output_filename

    # Exporter le JSON
    with open(output_file, "w") as f:
        json.dump(json_dataset_description, f, indent=4)

    hp2b_log.info(f"JSON exporté vers {output_dir}")
