# RODEO II Data Analysis Workflow

William Jenkins, Ph.D.  
Scripps Institution of Oceanography
University of California San Diego

This repository depends on the [`rodeo`](https://github.com/NeptuneProjects/RODEO-II) Python package.

## Workflow

### Configuration

This workflow makes use of a number of configuration files, data sources, etc.
Additionally, the outputs are numerous and varied.
Paths are specified in a centralized manner to make it easy to change the location of data files, outputs, etc., and can be found at `config/paths.toml`

Since the workflow is designed to be run either locally or on a remote server, computer-specific paths can be set in the `paths.toml` file.
To set this up, run the configuration script and specify an alias for the computer you're on:
```bash
python src/southfork/config.py set-id --alias=<your_alias>
```
The script will either:
- Create a new `.env` file in the root directory with your computer identifier and alias if one does not exist.
- Update the existing `.env` file with your computer identifier and alias if one already exists and the ID and alias are present.
- Append your computer identifier and alias to the existing `.env` file if one already exists and the ID and alias are not present.

To update the alias, run:
```bash
python src/southfork/config.py set-alias <new_alias>
```

Running the script will also show what paths are set for your computer and whether or not they exist. To view the paths set for your computer, run:
```bash
python src/southfork/config.py
```

With the alias set, the paths in `config/paths.toml` will be updated to point to the correct locations for your computer.
Ensure the alias is consistent with the headings in the TOML file.
For example, for three aliases, `work-laptop`, `home-desktop`, and `lab-server`, the paths file could look like this:
```toml
[computer.work-laptop]
data_dir = "D:/Projects/BigData"
output_dir = "E:/Results"

[computer.home-desktop]
data_dir = "C:/Users/me/Datasets"
output_dir = "D:/ProjectOutput"

[computer.lab-server]
data_dir = "/data/shared/datasets"
output_dir = "/home/user/results"
temp_dir = "/tmp/myproject"
```

If any keys in the computer-specific sections are the same as the default keys, the default values will be overridden.

### Construct an inventory of data files.

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
