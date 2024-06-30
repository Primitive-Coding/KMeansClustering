import os

import numpy as np

import datetime as dt

# Pandas
import pandas as pd
import pandas_datareader as web

# Sklearn
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Yahoo
from yahoo_fin import stock_info as si
import yfinance as yf

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt


class KMeansCluster:
    def __init__(
        self,
        n_clusters: int = 5,
        period: int = 5,
        period_unit: str = "Y",
        use_local_data: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        n_clusters: int
            Number of clusters.
        period : int
            Number of periods.
        period_unit : str
            The unit of the period. For example, if period=5, and period_unit="Y", then the full period will be 5 years.
        use_local_data : bool
            Determine if historical data is read and written locally to reduce Yahoo query times.
        """
        self.n_clusters = n_clusters
        self.period = period
        self.period_unit = period_unit
        self.use_local_data = use_local_data

    def create_historical_prices_cluster(self, ticker_list: list):
        """
        Create clusters based on historical daily price movements, based on a set of tickers.
        """
        data = self._fetch_externally(ticker_list)
        data.dropna(inplace=True)
        open_values = np.array(data["Open"].T)
        close_values = np.array(data["Close"].T)
        daily_movements = close_values - open_values
        # Logic to determine cluster sizes.
        try:
            sample_size = len(data["Adj Close"].columns.to_list())
        except AttributeError:
            sample_size = 1
        if sample_size < self.n_clusters:
            n_clusters = int(sample_size / 2)
        else:
            n_clusters = self.n_clusters
        # Create a normalizer to scale our data.
        normalizer = Normalizer()
        # Create clustering model & pipeline.
        clustering_model = KMeans(n_clusters=n_clusters, max_iter=1000)
        pipeline = make_pipeline(normalizer, clustering_model)
        try:
            pipeline.fit(daily_movements)
            clusters = pipeline.predict(daily_movements)

            results = pd.DataFrame(
                {"clusters": clusters, "tickers": ticker_list}
            ).sort_values(by=["clusters"], axis=0)

            results.set_index("tickers", inplace=True)

            return results
        except ValueError as e:
            print(
                f"[create_historical_prices_cluster() Error] Try *increasing* the quantity of tickers. Or *decrease* the 'n_clusters' value"
            )
            print(f"{e}")

    def _fetch_data(self, ticker_list: list):
        """
        Get the historical price data from yahoo finance.

        Parameters
        ----------
        ticker_list : list
            List of tickers to fetch data for.

        Returns
        -------
        pd.DataFrame
            DataFrame of historical prices for all the tickers in "ticker_list".
        """

        """ 
        Work In Progress
        """

        cwd = os.getcwd()
        # If the class is called from outside this project directory.
        external_path = f"{cwd}\\KMeansClustering\\Data\\historical_data.csv"
        # If the class is called within the project directory.
        internal_path = f"{cwd}\\Data\\historical_data.csv"
        try:
            try:

                df = pd.read_csv(external_path)
            except FileNotFoundError:
                df = pd.read_csv(internal_path)
                print(f"DF: {df}")
        except FileNotFoundError:
            print(f"TAGGGGG")
            df = self._fetch_externally(ticker_list)
            try:
                df.to_csv(external_path)
            except OSError:
                df.to_csv(internal_path)
            return df

    def _fetch_externally(self, ticker_list: list):
        delta = self._get_delta()
        start = dt.datetime.now() - delta
        end = dt.datetime.now()
        # data = web.DataReader(tickers, "yahoo", start, end)
        data = yf.download(tickers=ticker_list, start=start, end=end)
        return data

    def _get_delta(self):
        """
        Create a "timedelta" for date calculations.

        Parameters
        ----------
        period : int
            Number of periods.
        period_unit : str
            The unit of the period. For example, if period=5, and period_unit="Y", then the full period will be 5 years.

        Returns
        -------
        dt.timedelta
            Time delta with the adjusted amount of days according to the 'period_unit'.
        """
        if self.period_unit == "Y":
            return dt.timedelta(days=(365 * self.period))
        elif self.period_unit == "M":
            return dt.timedelta(days=(30 * self.period))
        elif self.period_unit == "D":
            return dt.timedelta(days=self.period)

    """--------------------------------------- DOW 30 Tickers ---------------------------------------"""

    def get_DOW_tickers(self):
        """
        Get tickers from the Dow Jones Industrial index.

        Returns
        -------
        list
            List of tickers within the index
        """
        df = self._read_dow30_file()
        index = df.index.to_list()
        return index

    def _read_dow30_file(self):
        cwd = os.getcwd()
        path = f"{cwd}\\KMeansClustering\\Tickers\\dow30.csv"
        try:
            df = pd.read_csv(path)
            df.rename(columns={"Unnamed: 0": "Symbol"}, inplace=True)
        except FileNotFoundError:
            path = f"{cwd}\\Tickers\\dow30.csv"
            df = pd.read_csv(path)
            df.rename(columns={"Unnamed: 0": "Symbol"}, inplace=True)
        print(f"Ticker: {df}")
        df.set_index("Symbol", inplace=True)
        return df

    """--------------------------------------- SP500 Tickers ---------------------------------------"""

    def get_SP500_tickers(self, yahoo_format: bool = True):
        """
        Get tickers from the S&P 500 index.

        Parameters
        ----------

        yahoo_format: bool
            Determines if tickers are formatted to work with Yahoo Finance Api.

        Returns
        -------
        list
            List of tickers within the index
        """

        df = self._read_sp500_file()
        index = df.index.to_list()
        if yahoo_format:
            j = 0
            for i in index:
                if "." in i:
                    index[j] = i.replace(".", "-")
                j += 1

        return index

    def _read_sp500_file(self):
        cwd = os.getcwd()
        path = f"{cwd}\\KMeansClustering\\Tickers\\sp500.csv"
        try:
            df = pd.read_csv(path)
            df.rename(columns={"Unnamed: 0": "Symbol"}, inplace=True)
        except FileNotFoundError:
            path = f"{cwd}\\Tickers\\sp500.csv"
            df = pd.read_csv(path)

        df.set_index("Symbol", inplace=True)
        return df

    def test(self):

        tickers = ["AAPL", "AMZN", "MSFT", "TSLA"]

        columns = pd.MultiIndex.from_product(
            [
                ["Adj Close", "Close", "High", "Low", "Open", "Volume"],
                tickers,
            ],
            names=["Attribute", "Ticker"],
        )
        # Generate some sample data
        dates = pd.date_range("2019-07-01", periods=10)
        data = np.random.rand(len(dates), len(columns))

        print(f"Data: {data}")


if __name__ == "__main__":
    k = KMeansCluster(n_clusters=2)
    k.test()
    # ticker_list = k.get_SP500_tickers()
    # ticker_list = ticker_list[:5]
    # # print(f"Tickers: {ticker_list}")
    # print(k._fetch_externally(ticker_list))
    # ticker_list = ticker_list[:50]

    # df = k.create_historical_prices_cluster(ticker_list)

    # print(f"DF: {df}")
