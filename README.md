# Machine Learning and Data Science Projects

Portfolio repository for Python, notebook, and data science projects.

## Repository Contents

- `projet_01_superstore_eda.py` - retail sales exploratory analysis with generated charts
- `projet_02_rfm_segmentation_1.py` - RFM customer segmentation analysis
- `data-science-projects/` - organized project folders for notebooks, scripts, and datasets
- `tests/` - basic Python source compile checks
- `environment.yml` - Conda environment definition

## Projects

- `data-science-projects/spy-market-analysis/` - SPY market analysis notebook
- `data-science-projects/time-series-household-power/` - household power consumption time-series work
- `data-science-projects/commodity-futures/` - commodity futures notebook and supporting data
- `data-science-projects/rfm-segmentation/` - RFM segmentation project and generated customer segment CSV

## Setup

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate my-projectsml
```

If the environment name changes, check the `name` field inside `environment.yml`.

## Run Checks

```bash
python -m pytest
```

## Data Notes

Some large datasets are intentionally kept out of Git. See `data-science-projects/README.md` for project-specific notes before running notebooks that expect local data files.

