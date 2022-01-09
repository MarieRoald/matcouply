import json
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests
from scipy.io import loadmat
from tqdm import tqdm

DATASET_PARENT = Path(__file__).parent / "datasets"
DOWNLOADED_PARENT = DATASET_PARENT / "downloads"


def get_bike_data():
    r"""Get bike sharing data from three major Norwegian cities

    This dataset contains three matrices with bike sharing data from Oslo, Bergen and Trondheim,
    :math:`\mathbf{X}^{(\text{Oslo})}, \mathbf{X}^{(\text{Bergen})}` and :math:`\mathbf{X}^{(\text{Trondheim})}`.
    Each row of these data matrices represent a station, and each column of the data matrices
    represent an hour in 2021. The matrix element :math:`x^{(\text{Oslo})}_{jk}` is the number of trips
    that ended in station :math:`j` in Oslo during hour :math:`k`.

    The data was obtained using the open API of
    
     * Oslo Bysykkel: https://oslobysykkel.no/en/open-data
     * Bergen Bysykkel: https://bergenbysykkel.no/en/open-data
     * Trondheim Bysykkel: https://trondheimbysykkel.no/en/open-data

    on the 23rd of November 2021.

    The dataset is cleaned so it only contains for the dates in 2021 when bike sharing was open in all three
    cities (2021-04-07 - 2021-11-23).

    Returns
    -------
    dict
        Dictionary mapping the city name with a data frame that contain bike sharing data from that city.
        There is also an additional ``"station_metadata"``-key, which maps to a data frame with additional
        station metadata. This metadata is useful for interpreting the extracted components.
    
    Note
    ----
    The original bike sharing data is released under a NLOD lisence (https://data.norge.no/nlod/en/2.0/).
    """
    with ZipFile(DATASET_PARENT / "bike.zip") as data_zip:
        with data_zip.open("bike.json", "r") as f:
            bike_data = json.load(f)

    datasets = {key: pd.DataFrame(value) for key, value in bike_data["dataset"].items()}
    time = [datetime(2021, 1, 1) + timedelta(hours=int(h)) for h in datasets["oslo"].columns]

    out = {}
    for city in ["oslo", "trondheim", "bergen"]:
        df = datasets[city]
        df.columns = time
        df.columns.name = "Time of arrival"
        df.index.name = "Arrival station ID"
        df.index = df.index.astype(int)
        out[city] = df

    out["station_metadata"] = datasets["station_metadata"]
    out["station_metadata"].index.name = "Arrival station ID"
    out["station_metadata"].index = out["station_metadata"].index.astype(int)

    return out


def get_semiconductor_etch_raw_data(download_data=True, save_data=True):
    """Load semiconductor etch data from :cite:p:`wise1999comparison`.

    If the dataset is already downloaded on your computer, then the local files will be
    loaded. Otherwise, they will be downloaded. By default, the files are downloaded from
    https://eigenvector.com/data/Etch.

    Parameters
    ----------
    download_data : bool
        If ``False``, then an error will be raised if the data is not
        already downloaded.
    save_data : bool
        if ``True``, then the data will be stored locally to avoid having to download
        multiple times.

    Returns
    -------
    dict
        Dictionary where the keys are the dataset names and the values are the contents
        of the MATLAB files.
    """
    data_urls = {
        "MACHINE_Data.mat": "http://eigenvector.com/data/Etch/MACHINE_Data.mat",
        "OES_DATA.mat": "http://eigenvector.com/data/Etch/OES_DATA.mat",
        "RFM_DATA.mat": "http://eigenvector.com/data/Etch/RFM_DATA.mat",
    }
    data_raw_mat = {}

    print("Loading semiconductor etch data from Wise et al. (1999) - J. Chemom. 13(3‐4), pp.379-396.")
    print("The data is available at: http://eigenvector.com/data/Etch/")
    for file, url in tqdm(data_urls.items()):
        file_path = DOWNLOADED_PARENT / file
        if file_path.exists():
            data_raw_mat[file] = loadmat(file_path)
        elif download_data:
            request = requests.get(url)
            if request.status_code != 200:
                raise RuntimeError(f"Cannot download file {url} - Response: {request.status_code} {request.reason}")

            if save_data:
                DOWNLOADED_PARENT.mkdir(exist_ok=True, parents=True)
                with open(file_path, "wb") as f:
                    f.write(request.content)

            data_raw_mat[file] = loadmat(BytesIO(request.content))
        else:
            raise RuntimeError("The semiconductor etch data is not yet downloaded, and ``download_data=False``.")
    return data_raw_mat


def get_semiconductor_etch_machine_data(download_data=True, save_data=True):
    """Load machine measurements from the semiconductor etch dataset from :cite:p:`wise1999comparison`.

    This function will load the semiconductor etch machine data and prepare it for analysis.

    If the dataset is already downloaded on your computer, then the local files will be
    loaded. Otherwise, they will be downloaded. By default, the files are downloaded from
    https://eigenvector.com/data/Etch.

    Parameters
    ----------
    download_data : bool
        If ``False``, then an error will be raised if the data is not
        already downloaded.
    save_data : bool
        if ``True``, then the data will be stored locally to avoid having to download
        multiple times.

    Returns
    -------
    dict
        Dictionary where the keys are the dataset names and the values are the contents
        of the MATLAB files.
    """
    # Get raw MATLAB data and parse into Python dict
    data = get_semiconductor_etch_raw_data(download_data=download_data, save_data=save_data)["MACHINE_Data.mat"][
        "LAMDATA"
    ]
    data = {key: data[key].squeeze().item().squeeze() for key in data.dtype.fields}

    # Format data as xarray dataset
    varnames = data["variables"][2:]

    # Get the training data
    train_names = [name.split(".")[0][1:] for name in data["calib_names"]]
    train_data = {
        name: pd.DataFrame(Xi[:-1, 2:], columns=varnames)  # Slice away last row since it "belongs" to the next sample
        for name, Xi in zip(train_names, data["calibration"])
    }
    train_metadata = {}
    for i, name in enumerate(list(train_data)):
        train_data[name].index.name = "Time point"
        train_data[name].columns.name = "Measurement"

        metadata = pd.DataFrame(data["calibration"][i][:-1, [0, 1]], columns=["Time", "Step number"])
        metadata["Experiment"] = int(name[:2])
        metadata["Sample"] = int(name[2:])
        metadata.index.name = "Time point"
        metadata.columns.name = "Metadata"
        train_metadata[name] = metadata

    # Get the testing data
    test_names = [name.split(".")[0][1:] for name in data["test_names"]]
    test_data = {
        name: pd.DataFrame(Xi[:-1, 2:], columns=varnames)  # Slice away last row since it "belongs" to the next sample
        for name, Xi in zip(test_names, data["test"])
    }
    test_metadata = {}
    for i, name in enumerate(list(test_data)):
        test_data[name].index.name = "Time point"
        test_data[name].columns.name = "Measurement"

        metadata = pd.DataFrame(data["test"][i][:-1, [0, 1]], columns=["Time", "Step number"])
        metadata["Experiment"] = int(name[:2])
        metadata["Sample"] = int(name[2:])
        metadata["Fault name"] = data["fault_names"][i]
        metadata.index.name = "Time point"
        metadata.columns.name = "Metadata"
        test_metadata[name] = metadata

    return train_data, train_metadata, test_data, test_metadata
