# Epsilon-Net Classifier for MAROH Data Analysis

## Project Overview

This project implements a classifier based on a pseudo ε-network for finding similar states in the MAROH multi-agent traffic flow balancing method. The system allows constructing a pseudo-ε network for training data compression, performing streaming classification, and visualizing the results.

## System Architecture

The project has a modular structure with clear separation of responsibilities:

project/
├── main.py # Entry point, configuration handling
├── experiment_runner.py # Experiment execution
├── data_loader.py # MAROH data loading
├── classifiers/ # Classifiers package
├── visualization/ # Visualization package

## Configuration System

The project uses a two-level configuration system with correct parameter priority:

1. **YAML configuration file** (required) — sets the base experiment parameters
2. **Command-line arguments** (optional) — override parameters from the config

**How it works:**
- Values from the config are loaded as base values
- Command-line arguments (if explicitly provided) override the corresponding parameters
- Only after this are default values applied for missing parameters

## Usage

### Basic run with config
python3 main.py --config config.yaml

### Run with parameter overrides
python3 main.py --config config.yaml --dist-percentile 5

### Command-line parameters
Parameter	        |Type	            |Description
|-------------------|-------------------|---------------------------------------------------|
--config	        |str	            |Required. Path to YAML configuration file
--data-path	        |str	            |Path to data (overrides value from config)
--mode	            |{single,full}	    |Execution mode
--agent	            |int	            |Agent ID (for single mode)
--iteration	        |int	            |Iteration number (for single mode)
--delta	            |float	            |Epsilon radius increase parameter [0, 1)
--theta	            |float	            |Classifier confidence threshold [0, 1]
--metric	        |str	            |Distance metric
--train-ratio	    |float	            |Proportion of data for training (0, 1)
--max-test-points	|int	            |Maximum number of test points
--use-acceleration	|flag	            |Use BallTree acceleration
--buffer-size	    |int	            |Buffer size for uncertain points
--output-dir	    |str	            |Directory for saving results
--verbose	        |flag	            |Verbose output
--seed	            |int	            |Seed for reproducibility
--save-plots	    |flag	            |Save plots

### Input Data
The configuration file provided in the repository loads train and test data from train4_1000 and test4 respectively.

### Output Data
After the experiment completes, the following files are saved in a timestamped subdirectory within the specified output directory:
Data files:
* analysis.json — complete results in JSON
* analysis.csv — summary table in CSV
* analysis_aggregated.json — aggregated metrics

Plots (if save_plots is enabled):
* heatmaps.png — heatmaps of metrics by agents and iterations
* distributions.png — distributions of accuracy, coverage, compression
* time_analysis.png — execution time analysis
* best_worst.png — best and worst results analysis

### System Requirements
* Python 3.8+
