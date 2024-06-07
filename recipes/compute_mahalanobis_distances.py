# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read recipe inputs
stock_returns = dataiku.Dataset("PCA_coordinates_scored_joined")
stock_returns_df = stock_returns.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
mahalanobis_distances = pd.DataFrame()

for cluster in stock_returns_df['cluster'].unique():
    pivot_stock_prices = pd.pivot_table(stock_returns_df[stock_returns_df['cluster']==cluster], 
                                        values='log_price_lag_diff',
                                        index='date', columns='ticker', aggfunc='last')


    pivot_stock_prices = pivot_stock_prices.dropna() # remove rows with missing values
    
    tickers = pivot_stock_prices.columns
    dates = pivot_stock_prices.index
    covariance_matrix_df = pivot_stock_prices.cov()
    
    mean_vector = np.matrix(pivot_stock_prices.mean(axis=0))
    inv_covmat = np.linalg.pinv(covariance_matrix_df)
    
    for i in range(pivot_stock_prices.shape[0]):
        returns = list(pivot_stock_prices.iloc[i])

        left_term = np.dot(returns - mean_vector, inv_covmat)
        mahal = np.dot(left_term, (returns - mean_vector).T)

        mahalanobis_distance = mahal[0, 0]
        mean_returns = np.mean(returns)
        for j in range(pivot_stock_prices.shape[1]):
            returns_without_stock = np.copy(returns)
            returns_without_stock[j] = mean_returns

            left_term = np.dot(returns_without_stock - mean_vector, inv_covmat)
            mahal = np.dot(left_term, (returns_without_stock - mean_vector).T)

            mahalanobis_distance_without_stock = mahal[0, 0]
            mahalanobis_distances = mahalanobis_distances.append(pd.DataFrame({'mahalanobis_distance': [mahalanobis_distance],
                                                                               'mahalanobis_distance_without': [mahalanobis_distance_without_stock],
                                                                               'ticker': [tickers[j]],
                                                                               'date': [dates[i]],
                                                                               'cluster': [cluster]}))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
mahalanobis_distances_output = dataiku.Dataset("mahalanobis_distances")
mahalanobis_distances_output.write_with_schema(mahalanobis_distances)