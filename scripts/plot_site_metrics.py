#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use(Path("config/jasa_fig_style.mplstyle"))

events = [1]

# ROW_KEYS = [
#     "SPL_rms",
#     "SPL_pk",
#     "SPL_pkpk",
#     "SEL",
#     "t_rise",
#     "t_decay",
#     "kurtosis",
#     "SEL_cum",
# ]


ROW_NAMES = {
    "SPL_rms": "$L_{p,\\mathrm{rms}}$\n(dB re 1 $\\mu$Pa)",
    "SPL_pk": "$L_{p,\\mathrm{pk}}$\n(dB re 1 $\\mu$Pa)",
    "SPL_pkpk": "$L_{p,\\mathrm{pk-pk}}$\n(dB re 1 $\\mu$Pa)",
    "SEL": "SEL",
    "t_rise": "t (rise)",
    "t_decay": "t (decay)",
    "kurtosis": "Kurtosis",
    "SEL_cum": "SEL (cum)",
}


STATIONS = [
    "Tetra",
    "VLA1",
    "VLA2",
]
YLIM_SCALE = 1.1

for event in events:

    fig, axs = plt.subplots(nrows=len(ROW_NAMES), ncols=len(STATIONS), figsize=(8, 10))

    for i, (row_key, row_name) in enumerate(ROW_NAMES.items()):

        min_val = np.inf
        max_val = 0.0

        for j, station in enumerate(STATIONS):

            index = np.arange(1, 10)
            data = np.arange(1, 10) * (i + 1)

            xlim_margin = 0.1 * (index.max() - index.min())
            xlim = (index.min() - xlim_margin, index.max() + xlim_margin)

            if max_val < max(data):
                max_val = max(data)
            if min_val > min(data):
                min_val = min(data)

            ax = axs[i, j]

            ax.plot(index, data)
            
            
            
            ax.grid()
            ax.set_xlim(xlim)

            # Control column titles
            if i == 0:
                ax.set_title(station)
            
            # Control row titles & ytick labels
            if j == 0:
                ax.set_ylabel(row_name)
            else:
                ax.set_yticklabels([])

            # Control x-axis ticks and labels
            if i == len(ROW_NAMES) - 1:
                ax.set_xlabel("Strike Index")
            else:
                ax.set_xticklabels([])
        
        # Control y-axis limits
        for j in range(len(STATIONS)):
            ylim_margin = 0.1 * (max_val - min_val)
            axs[i, j].set_ylim(min_val - ylim_margin, max_val + ylim_margin)
    

            
            



            # axs[ROW_KEYS.index(row_key)][STATIONS.index(station)].set_title(
            #     f"{station} {row_key}"
            # )

    plt.show()


    
