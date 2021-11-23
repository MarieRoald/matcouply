import json
from datetime import datetime, timedelta
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

DATASET_PARENT = Path(__file__).parent / "datasets"


def get_bike_data():
    r"""Get bike sharing data from three major Norwegian cities

    This dataset contain three matrices with bike sharing data from Oslo, Bergen and Trondheim,
    :math:`\mathbf{X}^{(\text{Oslo})}, \mathbf{X}^{(\text{Bergen})}` and :math:`\mathbf{X}^{(\text{Trondheim})}`.
    Each row of these data matrices represent a station, and each column of the data matrices
    represent an hour in 2021. The matrix element `x^{(\text{Oslo})}_{jk}` is the number of trips
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
