import pytest
from src.data.dataset import assert_no_patient_leakage


def test_disjoint_splits_pass():
    assert_no_patient_leakage({"train": [1,2], "val": [3], "test": [4]})


def test_overlap_fails():
    with pytest.raises(AssertionError):
        assert_no_patient_leakage({"train": [1,2], "val": [2], "test": [4]})

