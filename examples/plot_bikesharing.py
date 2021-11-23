"""
Detecting behavioural patterns in bike sharing data
---------------------------------------------------
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import tensorly as tl
from component_vis.factor_tools import factor_match_score
from wordcloud import WordCloud

import matcouply.cmf_aoadmm as cmf_aoadmm
from matcouply.coupled_matrices import CoupledMatrixFactorization
from matcouply.data import get_bike_data

###############################################################################
# Load the data
# ^^^^^^^^^^^^^
bike_data = get_bike_data()
matrices = [bike_data["oslo"].values, bike_data["bergen"].values, bike_data["trondheim"].values]

###############################################################################
# Fit non-negative PARAFAC2 models
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

all_models = []
all_errors = []
lowest_error = float("inf")
for init in range(5):
    print("-" * 50)
    print("Init:", init)
    print("-" * 50)
    out = cmf_aoadmm.parafac2_aoadmm(
        matrices,
        rank=4,
        non_negative=True,
        n_iter_max=1000,
        tol=1e-8,
        verbose=100,
        return_errors=True,
        random_state=init,
    )

    all_models.append(out[0])
    all_errors.append(out[3])

    if out[3][-1] < lowest_error:
        selected_init = init
        lowest_error = out[3][-1]

###############################################################################
# Check uniqueness of the NN-CMF models
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def get_stacked_CP_tensor(cmf):
    weights, factors = cmf
    A, B_is, C = factors

    stacked_cp_tensor = (weights, (A, np.concatenate(B_is, axis=0), C))
    return stacked_cp_tensor


print("Similarity with selected init")
for init, model in enumerate(all_models):
    if init == selected_init:
        print(f"Init {init} is the selected init")
        continue

    fms = factor_match_score(
        get_stacked_CP_tensor(model), get_stacked_CP_tensor(all_models[selected_init]), consider_weights=False
    )
    print(f"Similarity with selected init: {fms:}")


weights, (A, B_is, C) = all_models[selected_init]

###############################################################################
# Convert factor matrices to DataFrame
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We also sort the components by weight and clip the factors at 0 since the AO-ADMM algorithm
# may allow negative values with a very small magnitude

if weights is None:
    weights = 1

norms = np.linalg.norm(A, axis=0) * np.linalg.norm(B_is[0], axis=0) * np.linalg.norm(C, axis=0)
order = np.argsort(-weights * norms)

A = pd.DataFrame(np.maximum(0, A[:, order]), index=["Oslo", "Bergen", "Trondheim"])
B_is = [
    pd.DataFrame(np.maximum(0, B_is[0][:, order]), index=bike_data["oslo"].index),
    pd.DataFrame(np.maximum(0, B_is[1][:, order]), index=bike_data["bergen"].index),
    pd.DataFrame(np.maximum(0, B_is[2][:, order]), index=bike_data["trondheim"].index),
]
C = pd.DataFrame(np.maximum(0, C[:, order]), index=bike_data["oslo"].columns)

###############################################################################
# Plot the time-components
# ^^^^^^^^^^^^^^^^^^^^^^^^

C_melted = C.melt(value_name="Value", var_name="Component", ignore_index=False).reset_index()
fig = px.line(
    C_melted,
    x="Time of arrival",
    y="Value",
    facet_row="Component",
    hover_data={"Time of arrival": "|%a, %b %e, %H:%M"},
    color="Component",
)
fig

###############################################################################
# Analysing the time-mode
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# By briefly looking at the time-mode components, we immediately see that the fourth
# component displays behaviour during summer, when people in Norway typically have
# their vacation. If we zoom in a bit, we can see interesting behaviour for the first
# three components too. The first three components are the most active during week-days.
# The first component likely represents travel home from work, as it is active in the
# afternoon and the second component likely represents travel too work, as it is active
# in the morning. The third component however, is active the whole day, but mostly
# during the afternoon and the morning. Interestingly enough, the holiday component
# does is most active during weekends instead of week days.


###############################################################################
# Plot the strength of the components in each city
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A_melted = A.melt(value_name="Value", var_name="Component", ignore_index=False).reset_index()
A_melted["Component"] = A_melted["Component"].astype(str)  # Force discrete colormap
fig = px.bar(A_melted, x="index", y="Value", facet_row="Component", color="Component")
fig

###############################################################################
# Analysing the city-mode components
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We see that most of the components are most prominant in Oslo (which is the
# largest city too), except for the third component, which is mainly prominent
# in Bergen instead.

###############################################################################
# Plot the Oslo-station components as a density-map
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

B_0_melted = (
    B_is[0]
    .join(bike_data["station_metadata"])
    .melt(value_name="Value", var_name="Component", ignore_index=False, id_vars=bike_data["station_metadata"].columns)
    .reset_index()
)

fig = px.density_mapbox(
    B_0_melted,
    lat="Arrival station latitude",
    lon="Arrival station longitude",
    z="Value",
    zoom=11,
    opacity=0.5,
    animation_frame="Component",
    animation_group="Arrival station ID",
)
fig.update_layout(mapbox_style="carto-positron",)
fig


###############################################################################
# Plot the Bergen-station components as a density-map
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

B_1_melted = (
    B_is[1]
    .join(bike_data["station_metadata"])
    .melt(value_name="Value", var_name="Component", ignore_index=False, id_vars=bike_data["station_metadata"].columns)
    .reset_index()
)

fig = px.density_mapbox(
    B_1_melted,
    lat="Arrival station latitude",
    lon="Arrival station longitude",
    z="Value",
    zoom=11,
    opacity=0.5,
    animation_frame="Component",
    animation_group="Arrival station ID",
)
fig.update_layout(mapbox_style="carto-positron",)
fig


###############################################################################
# Plot the Trondheim-station components as a density-map
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

B_2_melted = (
    B_is[2]
    .join(bike_data["station_metadata"])
    .melt(value_name="Value", var_name="Component", ignore_index=False, id_vars=bike_data["station_metadata"].columns)
    .reset_index()
)

fig = px.density_mapbox(
    B_2_melted,
    lat="Arrival station latitude",
    lon="Arrival station longitude",
    z="Value",
    zoom=11,
    opacity=0.5,
    animation_frame="Component",
    animation_group="Arrival station ID",
)
fig.update_layout(mapbox_style="carto-positron",)
fig

###############################################################################
# Plot the station components as word-clouds
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

n_components = B_is[0].shape[1]
B_is_with_meta = [B_i.join(bike_data["station_metadata"]) for B_i in B_is]

fig, axes = plt.subplots(n_components, 3, figsize=(10, 2 * n_components), dpi=200, squeeze=False)

for r in range(n_components):
    for city_id in range(3):
        wc = WordCloud(background_color="black", max_words=1000, colormap="Pastel1")
        frequencies = B_is_with_meta[city_id].set_index("Arrival station name")[r].to_dict()
        wc.generate_from_frequencies(frequencies)
        axes[r, city_id].imshow(wc, interpolation="bilinear")
        axes[r, city_id].set_xticks([])
        axes[r, city_id].set_yticks([])

axes[0, 0].set_title("Oslo")
axes[0, 1].set_title("Bergen")
axes[0, 2].set_title("Trondheim")
for i in range(4):
    axes[i, 0].set_ylabel(f"Component {i}")
plt.show()
