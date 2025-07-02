from halfpipe2bids.utils import regex_to_regressor


def test_regex_to_regressor():
    regex_confounds = [
        "c_comp_cor_0[0-4]",
        "(trans|rot)_[xyz]",
        "global_signal",
        "motion_outlier[0-9]+",
    ]
    confounds_columns = [
        "c_comp_cor_00",
        "c_comp_cor_01",
        "a_comp_cor_00",
        "global_signal",
        "global_signal_derivative1",
        "white_matter",
        "trans_x",
        "trans_x_derivative1",
        "rot_y",
        "rot_x_derivative1",
        "motion_outlier1",
        "motion_outlier2",
    ]
    matched = regex_to_regressor(regex_confounds, confounds_columns)
    assert matched == [
        "c_comp_cor_00",
        "c_comp_cor_01",
        "global_signal",
        "trans_x",
        "rot_y",
        "motion_outlier1",
        "motion_outlier2",
    ]
