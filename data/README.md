# MovieLens Dataset

This folder should contain the MovieLens dataset files.

## How to get the data

Download from: https://grouplens.org/datasets/movielens/

### Option A — MovieLens 1M (smaller, faster training)
Place these files here:
- `movies.dat`
- `ratings.dat`
- `users.dat`

### Option B — MovieLens 25M (recommended, more accurate)
Place these files here:
- `movies.csv`
- `ratings.csv`

## Why are dataset files not included?

The dataset files are excluded from this repository because:
1. They are large (MovieLens 25M is ~250MB compressed)
2. They belong to GroupLens Research — please download directly from their site
3. Git is not designed to store large binary/CSV files

## Citation

F. Maxwell Harper and Joseph A. Konstan. 2015.
The MovieLens Datasets: History and Context.
ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 51:1–51:19.
https://doi.org/10.1145/2827872
