__VERSION__ = "v1.0"
__AUTHOR__ = ["Yu Liang"]

import datetime
import itertools
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from extract_data import ExtractData as ED


class RelevanceBasedPrediction:
    def __init__(self):
        pass
    
    def grid_prediction(self, df:pd.DataFrame, date_column:str, predictor_columns_list:list[str], response_column:str, target_idx:int, time_delta:int, r_thresholds:list[int], plot=False):
        predictor_combinations = []
        for r in range(1, len(predictor_columns_list) + 1):
            combinations = list(itertools.combinations(predictor_columns_list, r))
            predictor_combinations.extend(combinations)
        
        # Convert tuples into string format for column names
        column_names = ['_'.join(combo) for combo in predictor_combinations]
        
        weight_dict = {"r_threshold":[]} | {key: [] for key in column_names}
        prediction_dict = {"r_threshold":[]} | {key: [] for key in column_names}
        
        for r_threshold in r_thresholds:
            weight_dict["r_threshold"].append(r_threshold)
            prediction_dict["r_threshold"].append(r_threshold)
            
            for combo in predictor_combinations:
                predictor_columns = list(combo)
                prediction, fit, adjusted_fit, _ = self.get_prediction(
                    df=df,
                    date_column=date_column,
                    predictor_columns=predictor_columns,
                    response_column=response_column,
                    target_idx=target_idx,
                    time_delta=time_delta,
                    r_threshold=r_threshold,      # if r_threshold below 0, weights can be negative
                    #num_of_training_obv=100
                )
                
                weight_dict['_'.join(combo)].append(adjusted_fit)
                prediction_dict['_'.join(combo)].append(prediction)
                
        weight_df = pd.DataFrame(weight_dict).set_index("r_threshold")
        weight_df = (weight_df/weight_df.sum().sum()).fillna(0)
        
        prediction_df = pd.DataFrame(prediction_dict).set_index("r_threshold")
        
        #print(weight_df)
        #print(prediction_df)
        
        prediction = (weight_df*prediction_df).sum().sum()
        
        if plot == True:
            sns.heatmap(weight_df, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
            plt.tight_layout()
            plt.show()
            
        return prediction
    
    def get_prediction(self, df:pd.DataFrame, date_column:str, predictor_columns:list[str], response_column:str, prediction_date:datetime.date, ticker:str, ticker_column:str, time_delta:int, max_lookback:pd.Timedelta, r_threshold:int):
        """
        time_delta to only look at historical observations time_delta days before the current observation happened
        """
        
        df = df.copy()
        
        df[date_column] = pd.to_datetime(df[date_column])
        xt_df = df[(df[date_column] == pd.to_datetime(prediction_date)) & (df[ticker_column] == ticker)]
        print(xt_df)
        
        # Only use observations that happened before the target observation for prediction
        df = df[df[date_column] < xt_df[date_column].iloc[0] - pd.to_timedelta(time_delta, unit='d')]
        # Only use observations that happened before the target observation for prediction
        df = df[df[date_column] > xt_df[date_column].iloc[0] - max_lookback]
        
        df = df.sort_values(by=date_column, ascending=False)
        xt = xt_df[predictor_columns].to_numpy(dtype="float64")
        X = df[predictor_columns].to_numpy()
        
        relevance_scores, sim, info_t, info_scores = self.relevance_scores(X, xt)
        
        df["Similarity of current observation to target observation"] = sim
        df["Informativeness of target observation"] = float(info_t[0])
        df["Informativeness of current observation"] = info_scores
        df["Relevance Score"] = relevance_scores

        df = df.sort_values(by="Relevance Score", ascending=False)
        #weight from relevance based prediction paper
        df["adjusted weights"] = self.weights(relevance_df=df, relevance_column_name="Relevance Score", r_threshold=r_threshold)
        #print(df)
        
        df["gamma_r"] = df[["Relevance Score"]].map(lambda x: 1 if x > r_threshold else 0)
        df["weights"] = df["Relevance Score"]*df["gamma_r"]/(df["Relevance Score"]*df["gamma_r"]).sum()
        
        prediction = sum(df[response_column]*df["weights"])

        #highest_relevance_score = float(df["Relevance Score"].iloc[0])
        fit = self.fit(relevance_df=df, weights_column="adjusted weights", response_column=response_column)
        asymmetry = self.asymmetry(relevance_df=df, relevance_column_name="Relevance Score", r_threshold=r_threshold, weights_column="adjusted weights", response_column=response_column)
        adjusted_fit = len(predictor_columns)*(fit + asymmetry)
        if np.isnan(adjusted_fit):
            print("ERROR", adjusted_fit)
        
        return prediction, fit, adjusted_fit, df
    
    def relevance_scores(self, X:np.array, xt:np.array) -> tuple[np.array, np.array, np.array]:
        """
        info_i is informativeness of all prior observations
        info_t is informativeness of current observation
        """
        info_t = self.info_scores(X=X, xi=xt.reshape(1,-1))
        info_scores = self.info_scores(X=X, xi=X)
        
        sim = self.mahalanobis_distances(X, xt)
        
        relevance_scores = sim + 0.5*(info_t + info_scores)
        return relevance_scores, sim, info_t, info_scores
    
    def weights(self, relevance_df: pd.DataFrame, relevance_column_name: str, r_threshold: float):
        """
        w = 1/N + lambda^2/(n-1)(gamma(r)r-phi*mean(r_sub))
        """
        N = len(relevance_df)
        relevance_df["gamma_r"] = relevance_df[[relevance_column_name]].map(lambda x: 1 if x > r_threshold else 0)
        n = relevance_df["gamma_r"].sum()
        phi = n/N
        
        r_sub_mean = (1/n)*(relevance_df["gamma_r"]*relevance_df[relevance_column_name]).sum()
        
        lambda_square = (1/(N-1))*(relevance_df[relevance_column_name]**2).sum()/((1/(n-1))*(relevance_df["gamma_r"]*(relevance_df[relevance_column_name]**2)).sum())
        #print(N,n,phi,r_sub_mean, lambda_square)
        
        relevance_df["weights"] = 1/N + (lambda_square/(n-1))*(relevance_df["gamma_r"]*relevance_df[relevance_column_name]-phi*r_sub_mean)
        #print(relevance_df["weights"].sum())
        
        return relevance_df["weights"]
    
    def mahalanobis_distances(self, X:np.array, xt:np.array) -> np.array:
        """
        sim(xi, xt) = -1/2*(xi - xt)*cov_inv(X)*(xi - xt)' where
                
        X: Matrix of all prior observations, where each row represents xi
        xi is a row vector of prior observations
        xt is a 1D row vector of current observation
        Cov-1(X.T) is the inverse covariance matrix of X to capture the correlation of the attributes of the previous observations
        
        Outputs (1,n) matrix where n is number of rows in X
        """
        cov_inverse = self.cov_inverse(X)
        
        return -np.einsum('ij,ij->i', np.matmul(X - xt, cov_inverse), (X - xt))      # einsum calculates dot product of ith row on left with ith row on right
    
    def info_scores(self, X:np.array, xi:np.array) -> np.array:
        """
        xi is a 2D matrix where each row represents an observation to derive informativeness for
        info(xi, xmean) = (xi - xmean)*cov_inv(X)*(xi - xmean)'
        """
        cov_inverse = self.cov_inverse(X)
        xmean = np.mean(X, axis=0)
        
        return np.einsum('ij,ij->i', np.matmul(xi - xmean, cov_inverse), (xi - xmean))      # einsum calculates dot product of ith row on left with ith row on right

    def cov_inverse(self, X:np.array) -> np.array:
        if X.shape[1] == 1:
            cov = np.array([[np.cov(X.T)]])
        else:
            cov = np.cov(X.T)
            
        cov_inverse = np.linalg.inv(cov)    # X.T to capture the correlation of the attributes of the previous observations

        return cov_inverse
    
    def fit(self, relevance_df:pd.DataFrame, weights_column:str, response_column:str):
        fit = (relevance_df[weights_column].corr(relevance_df[response_column]))**2
        if np.isnan(fit):
            fit = 0
        return fit
    
    def asymmetry(self, relevance_df:pd.DataFrame, relevance_column_name:str, r_threshold:float, weights_column:str, response_column:str):
        w_plus_df = relevance_df[relevance_df[relevance_column_name] > r_threshold]
        w_plus_fit = self.fit(relevance_df=w_plus_df, weights_column=weights_column, response_column=response_column)
        
        w_minus_df = relevance_df[relevance_df[relevance_column_name] < r_threshold]
        w_minus_fit = self.fit(relevance_df=w_minus_df, weights_column=weights_column, response_column=response_column)
        
        return 0.5*(np.sqrt(w_plus_fit) - np.sqrt(w_minus_fit))**2
    
def demo():
    RBP = RelevanceBasedPrediction()
    df = ED().extract_data_for_prediction_by_group(days_into_future=60, fama_industry="Business Services", start_date=datetime.date(2014,11,13), end_date=datetime.date(2023,9,29))
    df["iv_error"] = df[f"iv60d"] - df[f"60_days_future_volatility"]
    print(df)
    
    response_column="iv_error"

    print("Prediction, fit, df", RBP.get_prediction(
        df=df,
        date_column="date",
        predictor_columns = ['iv10d', 'iv20d', 'iv30d', 'iv60d', 'iv90d', 'iv6m'],
        response_column=response_column,
        prediction_date=datetime.date(2023,9,28),
        ticker="KELYA",
        ticker_column="ticker",
        time_delta=60,
        max_lookback=pd.Timedelta(days=365),
        r_threshold=0,      # if r_threshold below 0, weights can be negative
        #num_of_training_obv=100
        ))
    
    #print("Grid Prediction, fit, df", RBP.grid_prediction(
    #    df=df,
    #    date_column="date",
    #    predictor_columns_list=["ev", "evebit", "evebitda"], #"marketcap", "pb", "pe", "ps"],
    #    response_column="10 days historical volatility",
    #    target_idx=index,
    #    time_delta=10,
    #    r_thresholds=[0, 0.5, 1],      # if r_threshold below 0, weights can be negative
    #    #num_of_training_obv=100,
    #    plot=True
    #    ))
    
if __name__ == "__main__":
    demo()