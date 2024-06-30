# KMeans Clustering

- Create clusters based on a collection of tickers.
- Repo comes with:
  - Tunable KMeans Model. Found in `kmeans_cluster.py`
  - Collection of S&P 500 tickers. Found in `Tickers/sp500.csv`.
  - Collection of Dow 30 tickers. Found in `Tickers/dow30.csv`.
  - Add your own collections in the folder `Tickers`.

---

### Setup

1. Clone git repository: `https://github.com/Primitive-Coding/KMeansClustering.git`

2. Install the projects requirements with `pip install -r requirements.txt`

---

### Instructions

- Create a class instance.

```
    k = KMeansCluster(n_clusters=5) # n_clusters = Number of clusters used.
```

###### Creating Clusters

```
    # Get a list of tickers
    ticker_list = k.get_SP500_tickers()

    df = k.create_historical_prices_cluster(ticker_list)

    ------ Output ------

                clusters
    tickers
    ZBH             0
    ABT             0
    APH             0
    AMGN            0
    AME             0
    ...           ...
    TXT             4
    AVGO            4
    AXON            4
    AON             4
    VRSN            4

    [503 rows x 1 columns]
```
