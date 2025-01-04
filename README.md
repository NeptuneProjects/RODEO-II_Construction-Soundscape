# RODEO II Data Analysis Workflow

William Jenkins, Ph.D.  
Scripps Institution of Oceanography
July 2024

## Workflow

Most paths are set in a `.env` file located in the root directory.
Exceptions are specified below, and are usually supplied as command-line arguments.

### 1. Construct a catalogue of data files.

1. Configure the dataset inventory using the file `config/inventory.toml`.
2. Build the dataset inventory by running:  
   `python scripts/inventory.py`

### 2. Detect & save pile driving pulses.

1. For each pile driving event, tune the detector algorithm interactively using the notebook `scripts/detector.ipynb`.
The goal of this notebook is to index the same detections across all channels.
This sometimes requires adjusting detector parameters on a per-channel or per-station basis.
Done correctly, each channel will have an equal number of detections such that the detections are synchronized across all channels.
2. The notebook outputs pre-formatted `TOML` text that can be copied and pasted into an event file located in `config/events`.
3. Run the script `scripts/extract_pulses.py` to save the detections to disk.
The event to be processed is specified by supplying the event configuration file in `config/events` as an argument to the script, e.g.:
`python scripts/extract_pulses.py --event=config/events/2024-07-01.toml`  
Note that this script also computes and saves numerous soundscape metrics for each detection.
Two files result:
   - An HDF5 file containing the raw detections and metrics, e.g.:  
  `results/events/A08_2023-07-06/detections_SHRU905_VLA1.hdf5`
   - A CSV file containing a summary of the metrics, e.g.:  
    `results/events/A08_2023-07-06/metrics_SHRU905_VLA1.csv`
