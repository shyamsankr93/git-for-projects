# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from sklearn.decomposition import PCA

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read recipe inputs
stock_prices_log_returns = dataiku.Dataset("stock_prices_log_returns")
stock_prices_log_returns_df = stock_prices_log_returns.get_dataframe()

tickers_filter = dataiku.Dataset("price_by_ticker")
tickers_filter_df = tickers_filter.get_dataframe()
tickers = tickers_filter_df["ticker"]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
stock_prices_filtered = stock_prices_log_returns_df[stock_prices_log_returns_df["ticker"].isin(tickers)]
stock_prices_filtered

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
pivot_stock_prices = pd.pivot_table(stock_prices_filtered, values='log_price_lag_diff',
                                    index='date', columns='ticker')

pivot_stock_prices

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
covariance_matrix_df = pivot_stock_prices.cov()
covariance_matrix_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Compute recipe outputs from inputs
pca = PCA(4)
pca.fit(covariance_matrix_df)
pca.explained_variance_ratio_

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
pca_coordinates_df = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2", "PC3", "PC4"])
pca_coordinates_df['ticker'] = covariance_matrix_df.columns
pca_coordinates_df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
computed_PCA_coordinates = dataiku.Dataset("computed_PCA_coordinates")
computed_PCA_coordinates.write_with_schema(pca_coordinates_df)