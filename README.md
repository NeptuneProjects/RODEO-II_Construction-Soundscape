# RODEO II Data Analysis Workflow

William Jenkins, Ph.D.  
Scripps Institution of Oceanography
University of California San Diego

This repository depends on the [`rodeo` Python package](https://github.com/NeptuneProjects/RODEO-II).

## Workflow

Most paths are set in a `.env` file located in the root directory.
Exceptions are specified below, and are usually supplied as command-line arguments.

### 1. Construct an inventory of data files.

1. Configure the dataset inventory using the file `config/inventory.toml`.
2. Build the dataset inventory by running:  
   `python scripts/inventory.py`  
   The inventory is performed for each station, and saved as:  
   `data/inventory/SHRU905_VLA1_FileInfo.csv`

### 2. Detect & save pile driving pulses.

1. For each pile driving event, tune the detector algorithm interactively using the notebook `scripts/detector.ipynb`.
The goal of this notebook is to index the same detections across all channels.
This sometimes requires adjusting detector parameters on a per-channel or per-station basis.
Done correctly, each channel will have an equal number of detections such that the detections are synchronized across all channels.
2. The notebook outputs pre-formatted `TOML` text that can be copied and pasted into an event file located in `config/events`.
3. Run the script `scripts/pulse_extraction.py` to save the detections to disk.
The event to be processed is specified by supplying the event configuration file in `config/events` as an argument to the script, e.g.:  
`python scripts/pulse_extraction.py --event=config/events/2024-07-01.toml`  
An HDF5 file is saved to disk, e.g.:  
`results/events/A08_2023-07-06/detections_SHRU905_VLA1.hdf5`

### 3. Compute soundscape metrics.

Numerous soundscape metrics are computed for each detection by running the following script:  
`python scripts/metrics.py --event=config/events/2024-07-01.toml`  
A CSV file containing a summary of the metrics, e.g.:  
`results/events/A08_2023-07-06/metrics_SHRU905_VLA1.csv`