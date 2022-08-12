# MIT License: Copyright (c) 2022, Marie Roald.
# See the LICENSE file in the root directory for full license text.

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tensorly as tl
from tensorly.testing import assert_array_equal

from matcouply import data
from matcouply.coupled_matrices import CoupledMatrixFactorization
from matcouply.decomposition import _cmf_reconstruction_error
from matcouply.testing import assert_allclose


def test_get_simple_simulated_data():
    simulated_data, cmf = data.get_simple_simulated_data()
    assert isinstance(simulated_data, list)
    assert isinstance(cmf, CoupledMatrixFactorization)

    # Check that same random state gets same data
    simulated_data1, cmf1 = data.get_simple_simulated_data(random_state=1)
    simulated_data2, cmf2 = data.get_simple_simulated_data(random_state=1)

    for X1, X2 in zip(simulated_data1, simulated_data2):
        assert_array_equal(X1, X2)

    assert_array_equal(cmf1[1][0], cmf2[1][0])  # check A
    assert_array_equal(cmf1[1][2], cmf2[1][2])  # check C

    for B_i1, B_i2 in zip(cmf1[1][1], cmf2[1][1]):
        assert_array_equal(B_i1, B_i2)

    # Check that different random state gets different data
    simulated_data3, cmf3 = data.get_simple_simulated_data(random_state=2)

    assert not np.all(tl.to_numpy(cmf1[1][0]) == tl.to_numpy(cmf3[1][0]))
    assert not np.all(tl.to_numpy(cmf1[1][2]) == tl.to_numpy(cmf3[1][2]))

    # Check that noise level is correct
    simulated_data_noise_02, cmf_noise_02 = data.get_simple_simulated_data(noise_level=0.2, random_state=2)
    data_strength = tl.norm(cmf_noise_02.to_tensor())
    error = _cmf_reconstruction_error(simulated_data_noise_02, cmf_noise_02) / data_strength
    assert error == pytest.approx(0.2)


def test_get_bike_data():
    bike_data = data.get_bike_data()

    # Check that data has correct keys
    cities = ["oslo", "trondheim", "bergen"]

    # Check that data has correct columns
    stations = set(bike_data["station_metadata"].index)
    for city in cities:
        assert city in bike_data
        assert bike_data[city].shape[1] == bike_data["oslo"].shape[1]
        assert all(bike_data[city].columns == bike_data["oslo"].columns)

        for station in bike_data[city].index:
            assert station in stations


def test_get_etch_data_prints_citation(capfd):
    data.get_semiconductor_etch_raw_data()
    assert "Wise et al. (1999) - J. Chemom. 13(3‐4)" in capfd.readouterr()[0]

    data.get_semiconductor_etch_machine_data()
    assert "Wise et al. (1999) - J. Chemom. 13(3‐4)" in capfd.readouterr()[0]


@pytest.fixture
def patch_dataset_parent(tmp_path):
    old_path = data.DATASET_PARENT
    old_download_parent = data.DOWNLOADED_PARENT
    data.DATASET_PARENT = tmp_path
    data.DOWNLOADED_PARENT = tmp_path / "downloads"
    yield old_path
    data.DATASET_PARENT = old_path
    data.DOWNLOADED_PARENT = old_download_parent


class MockSuccessfullRequest(MagicMock):
    status_code = 200
    content = b""


class MockUnsuccessfullRequest(MagicMock):
    status_code = 404
    content = b""


@patch("matcouply.data.requests.get", return_value=MockSuccessfullRequest())
@patch("matcouply.data.loadmat")
@patch("matcouply.data.BytesIO")
def test_get_etch_raw_data_downloads_correctly(bytesio_mock, loadmat_mock, get_mock, patch_dataset_parent):
    with pytest.raises(RuntimeError):
        data.get_semiconductor_etch_raw_data(download_data=False)

    # Check that it works once when we download but don't save
    data.get_semiconductor_etch_raw_data(save_data=False)

    # Check that it wasn't saved when save_data=False
    with pytest.raises(RuntimeError):
        data.get_semiconductor_etch_raw_data(download_data=False)

    # Check that it raises error with unsuccessful download
    get_mock.return_value = MockUnsuccessfullRequest()
    with pytest.raises(RuntimeError):
        data.get_semiconductor_etch_raw_data()

    # Check that it works once the data is downloaded
    get_mock.return_value = MockSuccessfullRequest()
    data.get_semiconductor_etch_raw_data()
    data.get_semiconductor_etch_raw_data(download_data=False)


def test_get_semiconductor_etch_raw_data():
    raw_data = data.get_semiconductor_etch_raw_data()
    for file in ["MACHINE_Data.mat", "OES_DATA.mat", "RFM_DATA.mat"]:
        assert file in raw_data


def test_get_semiconductor_machine_data():
    train_data, train_metadata, test_data, test_metadata = data.get_semiconductor_etch_machine_data()

    train_sample_names = set(train_data)
    test_sample_names = set(test_data)
    train_metadata_names = set(train_metadata)
    test_metadata_names = set(test_metadata)

    assert len(train_sample_names.intersection(test_sample_names)) == 0
    assert len(train_sample_names.intersection(train_metadata_names)) == len(train_sample_names)
    assert len(test_sample_names.intersection(test_metadata_names)) == len(test_sample_names)

    # Check that data has correct columns
    one_train_name = next(iter(train_sample_names))
    for name in train_data:
        assert train_data[name].shape[1] == train_data[one_train_name].shape[1]
        assert train_metadata[name].shape[0] == train_data[name].shape[0]
        assert all(train_data[name].columns == train_data[one_train_name].columns)

    one_test_name = next(iter(test_sample_names))
    for name in test_data:
        assert test_data[name].shape[1] == test_data[one_test_name].shape[1]
        assert test_metadata[name].shape[0] == test_data[name].shape[0]
        assert all(test_data[name].columns == test_data[one_test_name].columns)
