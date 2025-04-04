__VERSION__ = "v1.0"
__AUTHOR__ = ["Yu Liang"]

import concurrent.futures
import datetime
import multiprocessing
import os
from typing import Literal

import matplotlib

matplotlib.use('Agg')  # This will use the Agg backend, which is non-GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import anderson, t
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

from extract_data import ExtractData as ED
from relevance_based_prediction.RelevanceBasedPrediction import \
    RelevanceBasedPrediction as RBP


class EvaluateRBP:
    def __init__(self):
        pass

    def process_ticker(self, df, ticker, ticker_column, predictor_columns, date_column, testing_size, days_into_future, max_lookback, model, mode, response_column, normal_experiment, r_threshold, plot_iv_figures, folder_path):
        try:
            predictor_columns = predictor_columns
            predicted_vs_actual_df = self.get_predicted_vs_actual_df(
                df=df,
                ticker=ticker,
                ticker_column=ticker_column,
                predictor_columns=predictor_columns,
                date_column=date_column,
                testing_size=testing_size,
                days_into_future=days_into_future,
                max_lookback=max_lookback,
                model=model,
                response_column=response_column,
                normal_experiment=normal_experiment,
                r_threshold=r_threshold,
                plot_iv_errors_fig=plot_iv_figures,
                folder_path=folder_path
            )
            
            rbp_rmse, iv_rmse, predicted_corr, iv_corr, pred_win_rate_over_iv, iv_adj_error_pred_df = self.test(
                ticker=ticker,
                predicted_vs_actual_df=predicted_vs_actual_df,
                days_into_future=days_into_future,
                model=model,
                mode=mode,
                folder_path=folder_path,
                normal_experiment=normal_experiment
            )
            
            return ticker, rbp_rmse, iv_rmse, predicted_corr, iv_corr, pred_win_rate_over_iv, iv_adj_error_pred_df
        except Exception as e:
            print(f"Error occurred for {ticker}: {e}")
            return ticker, None, None, None, None, None, None

    def batch_test(self, df, tickers_list: list[str], ticker_column, predictor_columns, date_column, testing_size, days_into_future, max_lookback, model, mode, response_column, normal_experiment, r_threshold, plot_iv_figures, folder_path="relevance_based_prediction/analysis/business_services_large_cap"):
        results_dict = {'ticker': [], 'rbp_rmse': [], 'iv_rmse': [], 'predicted_corr': [], 'iv_corr': [], 'pred_win_rate_over_iv': []}
        
        with multiprocessing.get_context("spawn").Pool(processes=os.cpu_count() or 1) as pool:
            results = pool.starmap(self.process_ticker, [(df, ticker, ticker_column, predictor_columns, date_column, testing_size, days_into_future, max_lookback, model, mode, response_column, normal_experiment, r_threshold, plot_iv_figures, folder_path) for ticker in tickers_list])
        
        for ticker, rbp_rmse, iv_rmse, predicted_corr, iv_corr, pred_win_rate_over_iv, iv_adj_error_pred_df in results:
            results_dict['ticker'].append(ticker)
            results_dict['rbp_rmse'].append(rbp_rmse)
            results_dict['iv_rmse'].append(iv_rmse)
            results_dict['predicted_corr'].append(predicted_corr)
            results_dict['iv_corr'].append(iv_corr)
            results_dict['pred_win_rate_over_iv'].append(pred_win_rate_over_iv)
            #results_dict['iv_adj_error_pred_df'].append(iv_adj_error_pred_df)
        
        pd.DataFrame(results_dict).to_csv(f"{folder_path}/rmse.csv", index=False)
    
    def test(self, ticker, predicted_vs_actual_df, days_into_future:int=10, model=Literal["RBP", "Grid", "LinReg", "GARCH"], mode=Literal["iv", "iv_errors"], folder_path=f"relevance_based_prediction/analysis/business_services_large_cap", normal_experiment=True):
        """
        test performance of models against the actual values, evaluated across a few different metrics
        """
        
        if normal_experiment == True:
            plot_CI = True
        else:
            plot_CI = False
        
        if mode == "iv":
            self.plot_predicted_vs_actual_graph(ticker=ticker, predicted_vs_actual_df=predicted_vs_actual_df, model=model, days_into_future=days_into_future, response_column=f"{days_into_future}_days_future_volatility", compare_with_iv=True, title=f'Plot of {ticker} future realised volatility and predicted volatility', y_label_1="volatility", folder_path=folder_path, file_name=f'{ticker}.png')
        elif mode == "iv_errors":
            
            # plot of iv_errors prediction
            self.plot_predicted_vs_actual_graph(ticker=ticker, predicted_vs_actual_df=predicted_vs_actual_df, model=model, days_into_future=days_into_future, response_column="iv_error", compare_with_iv=False, title=f"Plot of {ticker} predicted vs actual iv errors", y_label_1="iv errors", folder_path=folder_path, file_name = f'{ticker}_iv_errors.png', plot_CI=plot_CI)
            
            # plot of iv prediction after adjusting iv for errors
            iv_adj_error_pred_df = predicted_vs_actual_df.copy()
            iv_adj_error_pred_df["predicted_iv_error"] = iv_adj_error_pred_df["prediction"]
            iv_adj_error_pred_df["prediction"] = iv_adj_error_pred_df[f"iv{days_into_future}d"] - iv_adj_error_pred_df["predicted_iv_error"]
            
            pred_win_rate_over_iv = len(
                iv_adj_error_pred_df[
                    abs(iv_adj_error_pred_df["prediction"] - iv_adj_error_pred_df[f"{days_into_future}_days_future_volatility"])
                    < abs(iv_adj_error_pred_df[f"iv{days_into_future}d"] - iv_adj_error_pred_df[f"{days_into_future}_days_future_volatility"])]) / len(iv_adj_error_pred_df)
            
            rbp_rmse, iv_rmse, predicted_corr, iv_corr = self.plot_predicted_vs_actual_graph(ticker=ticker, predicted_vs_actual_df=iv_adj_error_pred_df, model=model, days_into_future=days_into_future, response_column=f"{days_into_future}_days_future_volatility", compare_with_iv=True, title=f'Plot of {ticker} future realised volatility and predicted volatility', y_label_1="volatility", folder_path=folder_path, file_name=f'{ticker}_iv_adj.png', plot_CI=plot_CI, pred_win_rate_over_iv=pred_win_rate_over_iv)
            print(iv_adj_error_pred_df)
            
            return rbp_rmse, iv_rmse, predicted_corr, iv_corr, pred_win_rate_over_iv, iv_adj_error_pred_df
            
        return
        
    def get_predicted_vs_actual_df(self, df, ticker, predictor_columns, ticker_column, date_column, testing_size=0.25, days_into_future:int=10, max_lookback=pd.Timedelta(days=252), response_column:str="10_days_future_volatility", model=Literal["RBP", "Grid", "LinReg", "GARCH"], normal_experiment=False, plot_iv_errors_fig=0, r_threshold=0, folder_path=f'relevance_based_prediction/analysis/graphs'):
        """
        df: initialised dataframe containing all relevant data, predictor and response columns must be included within
        days_into_future: days into the future to predict volatility and compare against actual realised volatility
        """        

        print("Predictor columns:", predictor_columns)
        
        # get latest 25% of dates
        # Step 1: Get unique dates and sort them
        ticker_df = df[df[ticker_column]==ticker]
        unique_dates = ticker_df['date'].drop_duplicates().sort_values()

        # Step 2: Select the latest 25% of dates
        testing_dates = unique_dates.iloc[int(len(unique_dates) * (1-testing_size)):]  # Last 25% of dates

        print(testing_dates)
        
        df = df.sort_values(by=date_column, ascending=False).reset_index()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
            args = [(df, date_column, predictor_columns, response_column, prediction_date, ticker, ticker_column, max_lookback, days_into_future, r_threshold, folder_path, normal_experiment, plot_iv_errors_fig) for prediction_date in testing_dates]
            pred_results = list(executor.map(lambda p: self.predict_rbp(*p), args))
            
            #elif model == "Grid":
            #    prediction = rbp.grid_prediction(
            #        df=df,
            #        date_column=date_column,
            #        predictor_columns_list=predictor_columns,
            #        response_column=response_column,
            #        target_idx=target_idx,
            #        time_delta=30,
            #        r_thresholds=[0, 0.25, 0.5],
            #    )
            #    pred_df_dict["prediction"].append(prediction)
            #    pred_df_dict["fit"].append(1)
            #    pred_df_dict["date"].append(df[date_column][target_idx])  # Replaced "date" with date_column
            
            #elif model == "LinReg":
            #    #predictor_columns = predictor_columns
            #    predictor_columns = predictor_columns
            #    
            #    df[date_column] = pd.to_datetime(df[date_column])  # Use date_column here
            #    xt_df = df[(df[date_column] == pd.to_datetime(prediction_date)) & (df[ticker_column] == ticker)]
            #    train_df = df[df[date_column] < xt_df[date_column] - pd.to_timedelta(days_into_future, unit='d')].copy()
            #    #train_df = train_df.iloc[:90]
            #    X_train = train_df[predictor_columns].to_numpy()
            #    y_train = train_df[response_column].to_numpy()
            #    X_test = xt_df[predictor_columns].to_numpy().reshape(1, -1)
            #    #y_test = X_test.pop(response_column)
            #    
            #    linreg = LinearRegression()
            #    linreg.fit(X_train, y_train)
#
            #    prediction = linreg.predict(X_test)[0]
            #    pred_df_dict["prediction"].append(prediction)
            #    pred_df_dict["date"].append(prediction_date)  # Replaced "date" with date_column
            #    
            #    y_train_pred = linreg.predict(X_train)
            #    r2 = r2_score(y_train, y_train_pred)
            #    # Adjusted R^2 calculation
            #    n = len(y_train)  # Number of observations
            #    k = X_train.shape[1]  # Number of predictors
            #    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
            #    
            #    pred_df_dict["fit"].append(adjusted_r2)
            #
            #elif model == "GARCH":
            #    # Function to fit the GARCH model for each combination of p and q
            #    def fit_garch(p, q, y_train, prediction_date, df):
            #        try:
            #            # Fit GARCH(p,q) model
            #            garch = arch_model(y_train, vol="Garch", p=p, q=q, mean="Constant", dist="normal")
            #            results = garch.fit(disp="off")
#
            #            # Forecast variance for the next 10 days
            #            forecast = results.forecast(horizon=10)
            #            forecasted_variances = forecast.variance.values[-1, :]  # Variances for horizons 1 to 10
#
            #            # Aggregate variance to get total volatility over the next 10 days
            #            total_variance = np.sum(forecasted_variances)  # Sum variances for horizons 1 to 10
            #            aggregate_volatility = np.sqrt(total_variance) / 100  # Convert to volatility (standard deviation)
            #            print(aggregate_volatility)
#
            #            # Store the best model parameters and results
            #            return {
            #                'aic': results.aic,
            #                'order': (p, q),
            #                'model': results,
            #                'aggregate_volatility': aggregate_volatility,
            #                'date': prediction_date  # Replaced "date" with date_column
            #            }
            #        except Exception as e:
            #            print(f"Error fitting model with p={p}, q={q}: {e}")
            #            return None
#
            #    # Define ranges for p and q
            #    p_values = [1, 5, 10]
            #    q_values = [1, 5, 10]
#
            #    best_aic = float('inf')
            #    best_order = None
            #    best_model = None
            #    best_aggregate_volatility = None
            #    best_date = None
#
            #    # Convert dates and calculate returns
            #    df[date_column] = pd.to_datetime(df[date_column])  # Use date_column here
            #    df['returns'] = ((df['close'] - df['close'].shift(-1)) / df['close'].shift(-1)) * 100
#
            #    # Target date for prediction
            #    xt_df = df[(df[date_column] == pd.to_datetime(prediction_date)) & (df[ticker_column] == ticker)]
            #    train_df = df[df[date_column] < xt_df[date_column]].copy()  # Replaced "date" with date_column
            #    train_df = train_df.iloc[:90]
#
            #    # Historical returns for training
            #    y_train = train_df["returns"].dropna().to_numpy()
#
            #    # Initialize ThreadPoolExecutor to run GARCH fitting in parallel
            #    with ThreadPoolExecutor(max_workers=10) as executor:
            #        futures = []
            #        
            #        # Loop through combinations of p and q and submit each task to the executor
            #        for p in p_values:
            #            for q in q_values:
            #                futures.append(executor.submit(fit_garch, p, q, y_train, prediction_date, df))
#
            #        # Process results as they complete
            #        for future in as_completed(futures):
            #            result = future.result()
            #            if result and result['aic'] < best_aic:
            #                best_aic = result['aic']
            #                best_order = result['order']
            #                best_model = result['model']
            #                best_aggregate_volatility = result['aggregate_volatility']
            #                best_date = result['date']
#
            #    # Append results for the best model
            #    if best_model is not None:
            #        pred_df_dict["prediction"].append(best_aggregate_volatility)
            #        pred_df_dict["date"].append(best_date)
            #        pred_df_dict["fit"].append(best_aic)
        
        pred_df = pd.DataFrame(pred_results)
        print('pred df', pred_df)
        print(pred_df["date"].duplicated().sum())  # Check if `pred_df["date"]` has duplicates
        print(ticker_df[date_column].duplicated().sum())  # Check if `ticker_df[date_column]` has duplicates

        predicted_vs_actual_df = pd.concat([pred_df.set_index("date"), ticker_df.set_index(date_column)], axis=1, join="inner").reset_index()

        return predicted_vs_actual_df
    
    def plot_predicted_vs_actual_graph(self, ticker, predicted_vs_actual_df, model, days_into_future, response_column, compare_with_iv=True, date_column="date", y_label_1="Volatility", title='Plot of future realised volatility and predicted volatility', folder_path=f"relevance_based_prediction/analysis/graphs", file_name='ticker.png', plot_CI=False, pred_win_rate_over_iv=None):
        fig, ax1 = plt.subplots()

        # Plot with 'date' as the x-axis
        if plot_CI == True:
            ax1.fill_between(predicted_vs_actual_df[date_column], predicted_vs_actual_df['prediction'] - predicted_vs_actual_df["margin of error"], predicted_vs_actual_df['prediction'] + predicted_vs_actual_df["margin of error"], color="r", alpha=0.2, label=f"{model} CI (95%)")
                
        ax1.plot(predicted_vs_actual_df[date_column], predicted_vs_actual_df['prediction'], color="r", label=model)
        ax1.plot(predicted_vs_actual_df[date_column], predicted_vs_actual_df[response_column], color="g",label='actual')
        if compare_with_iv == True:
            ax1.plot(predicted_vs_actual_df[date_column], predicted_vs_actual_df[f'iv{days_into_future}d'], color="black",label=f'iv{days_into_future}d', alpha=0.5)
        #ax1.set_ylim(0, 0.3)
        
        ax2 = ax1.twinx()
        ax2.plot(predicted_vs_actual_df[date_column], predicted_vs_actual_df['fit'], color="b",label='fit', alpha=0.1)

        # Adding labels and title
        ax1.set_ylabel(y_label_1)
        ax2.set_ylabel('Fit')
        plt.title(title)
        plt.xlabel('Date')
        plt.xticks(rotation=45)
        
        # Add a legend
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        fig.subplots_adjust(bottom=0.30)

        graph_text, rbp_rmse, iv_rmse, predicted_corr, iv_corr = self.evaluation_metrics_text(predicted_vs_actual_df=predicted_vs_actual_df, days_into_future=days_into_future, response_column=response_column, compare_with_iv=compare_with_iv, pred_win_rate_over_iv=pred_win_rate_over_iv)
        # Add text with a transparent background below the graph
        fig.text(
            0.5, 0.02,  # (x, y) in figure coordinates (0=bottom, 1=top)
            graph_text,
            ha="center", fontsize=8,
            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5')  # Transparent box
        )
        
        
        # Create the folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)
        
        # Show the plot
        plt.savefig(f'{folder_path}/{file_name}')
        plt.close()
        
        return rbp_rmse, iv_rmse, predicted_corr, iv_corr
        
    def plot_iv_errors(self, data, date, max_lookback, r_threshold, folder_path, ticker, norm_res):
        fig, ax1 = plt.subplots()
        
        counts, bins = np.histogram(data, bins=30)  # Compute histogram counts and bin edges
        percentages = (counts / counts.sum()) * 100  # Convert frequencies to percentages
        
        plt.bar(bins[:-1], percentages, width=np.diff(bins), alpha=0.5, color='blue', edgecolor='black')
        plt.xlabel("iv_error")
        plt.ylabel('Percentage (%)')  # Changed from 'Frequency' to 'Percentage'
        plt.title('IV error in RBP subset')
        plt.ylim(0, 60)
        
        fig.subplots_adjust(bottom=0.30)
        
        # Compute mean and standard deviation
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        plt.xlim(mean_val-30, mean_val+30)
        
        # Prepare text for the box
        graph_text = f"Mean: {mean_val:.2f}\nStd Dev: {std_val:.2f}\nSize: {len(data)}\nNormal test result: {norm_res}\nMax lookback: {max_lookback.days} days\nr threshold: {r_threshold}"

        # Add text box below the graph
        fig.text(
            0.5, 0.02,  # (x, y) in figure coordinates (0=bottom, 1=top)
            graph_text,
            ha="center", fontsize=10,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')  # Box around text
        )

        # Create the folder if it doesn't exist
        os.makedirs(f'{folder_path}/{ticker}', exist_ok=True)
        
        # Show the plot
        plt.savefig(f'{folder_path}/{ticker}/{ticker}-{date}.png')
        plt.close()
            
    def evaluation_metrics_text(self, predicted_vs_actual_df, days_into_future, response_column, compare_with_iv=True, pred_win_rate_over_iv=None):
        actual_mean = round(predicted_vs_actual_df[response_column].mean(),2)
        print("ACTUAL MEAN", round(actual_mean,2))
        
        predicted_vs_actual_df = predicted_vs_actual_df.copy().dropna(subset=["prediction", response_column, f'iv{days_into_future}d'])
        predicted_mse = round(mean_squared_error(predicted_vs_actual_df[response_column], predicted_vs_actual_df['prediction']), 2)
        predicted_corr = round(predicted_vs_actual_df[response_column].corr(predicted_vs_actual_df['prediction']), 2)
        
        rbp_rmse = round(np.sqrt(predicted_mse), 2)
        
        graph_text = f"""
        Actual Mean: {actual_mean}
        
        RBP Evaluation
        MSE:{predicted_mse}, RMSE:{rbp_rmse}, Normalised RMSE: {round(np.sqrt(predicted_mse)/actual_mean*100, 2)}%, Correlation with actual: {predicted_corr}
        """
        
        if compare_with_iv == True:
            iv_mse = round(mean_squared_error(predicted_vs_actual_df[response_column], predicted_vs_actual_df[f'iv{days_into_future}d']), 2)
            iv_corr = round(predicted_vs_actual_df[response_column].corr(predicted_vs_actual_df[f'iv{days_into_future}d']), 2)
            
            iv_rmse = round(np.sqrt(iv_mse), 2)
            
            graph_text += f"""
            IV Evaluation
            MSE:{iv_mse}, RMSE:{iv_rmse}, Normalised RMSE: {round(np.sqrt(iv_mse)/actual_mean*100, 2)}%, Correlation with actual: {iv_corr}
            """
            
            graph_text += f"\nPrediction win rate over RBP: {round(pred_win_rate_over_iv*100,2)}%"
        
            return graph_text, rbp_rmse, iv_rmse, predicted_corr, iv_corr
        
        return graph_text, 0, 0, 0, 0
        
    def predict_rbp(self, df, date_column, predictor_columns, response_column, prediction_date, ticker, ticker_column, max_lookback, days_into_future, r_threshold, folder_path, normal_experiment=False, plot_iv_errors_fig=False):
        def get_filtered_data(df, r_threshold):
            """Filters the weight dataframe and removes outliers."""
            norm_df = df[df["weights"] > r_threshold]
            
            # filter outliers
            #Q1 = norm_df['iv_error'].quantile(0.25)
            #Q3 = norm_df['iv_error'].quantile(0.75)
            #IQR = Q3 - Q1
            #norm_df = norm_df[(norm_df['iv_error'] >= Q1 - 1.5 * IQR) & (norm_df['iv_error'] <= Q3 + 1.5 * IQR)]
            
            return norm_df


        def evaluate_normality(norm_df):
            """Evaluates normality using Anderson-Darling test."""
            data = norm_df['iv_error'].dropna()
            return round(anderson(data, dist="norm").statistic, 2), data


        def make_prediction(lookback, adj_threshold):
            return RBP().get_prediction(
                df=df,
                date_column=date_column,
                predictor_columns=predictor_columns,
                response_column=response_column,
                prediction_date=prediction_date,
                ticker=ticker,
                ticker_column=ticker_column,
                max_lookback=lookback,
                time_delta=days_into_future,
                r_threshold=adj_threshold,
            )

        pred_df_dict = {}

        # Initial prediction
        prediction, fit, adjusted_fit, weight_df = make_prediction(lookback=max_lookback, adj_threshold=r_threshold)
        norm_df = get_filtered_data(weight_df, r_threshold)
        norm_res, data = evaluate_normality(norm_df)

        # Try up to 2 more times if normality fails
        attempt = 0
        adj_max_lookback = max_lookback
        adj_r_threshold = r_threshold

        while norm_res < 10 and attempt < 3:
            #adj_max_lookback += pd.Timedelta(days=60)
            adj_r_threshold = round(adj_r_threshold - 0.2, 2)
            prediction, fit, adjusted_fit, weight_df = make_prediction(lookback=adj_max_lookback, adj_threshold=adj_r_threshold)
            norm_df = get_filtered_data(weight_df, r_threshold)
            norm_res, data = evaluate_normality(norm_df)
            attempt += 1
        
        # Calculate mean and margin of error
        mean = norm_df[response_column].mean()
        std = norm_df[response_column].std()
        confidence_level = 0.95
        n = len(norm_df)
        t_score = t.ppf(1 - (1 - confidence_level) / 2, df=n-1)  # t-value with df = n-1
        margin_of_error = t_score * std  # Uses sample standard deviation

        # Store results
        if normal_experiment == True:
            pred_df_dict["prediction"] = mean
            pred_df_dict["margin of error"] = margin_of_error
        else:
            pred_df_dict["prediction"] = prediction
            
        #if norm_res < 10:
        #    #norm_df = weight_df.sort_values(by="Relevance Score", ascending=False).iloc[:200]
        #    #norm_res, data = evaluate_normality(norm_df)
        #    
        #    pred_df_dict["prediction"] = prediction
        #    pred_df_dict["margin of error"] = 0
        
        pred_df_dict["fit"] = fit
        pred_df_dict["date"] = prediction_date

        if plot_iv_errors_fig == True:
            # Plot if needed
            self.plot_iv_errors(data=data, date=prediction_date.strftime("%Y-%m-%d"), max_lookback=adj_max_lookback, r_threshold=adj_r_threshold, folder_path=folder_path, ticker=ticker, norm_res=norm_res)
        return pred_df_dict
        
def init_df():
    price_df = pd.read_csv("relevance_based_prediction/data/NASDAQ_Daily_Metrics_INTC.csv")
    price_df["date"] = pd.to_datetime(price_df["date"])
    price_df = price_df.set_index("date")
    
    vol_df = pd.read_csv("relevance_based_prediction/data/aapl_volatility.csv")
    vol_df["date"] = pd.to_datetime(vol_df["date"])
    vol_df = vol_df.set_index("date")
    
    macro_df = pd.read_csv("relevance_based_prediction/data/macro/macro.csv")
    macro_df["DATE"] = pd.to_datetime(macro_df["DATE"])
    macro_df = macro_df.set_index("DATE").sort_index(ascending=True).rename(index={"DATE":"date"})
    macro_df = macro_df.ffill().dropna()
    
    imp_vol_df = pd.read_csv("relevance_based_prediction/data/filtered_INTC_data.csv")[["tradedate", "iv10d", "iv20d", "iv30d", "iv60d", "iv90d"]]
    imp_vol_df["tradedate"] = pd.to_datetime(imp_vol_df["tradedate"], format="%d/%m/%Y")
    imp_vol_df = imp_vol_df.set_index("tradedate").rename(index={"tradedate":"date"})
    
    df = pd.concat([vol_df], axis=1, join="inner")
    df.index.name = "date"
    print(df)
    df = df.sort_index(ascending=False).reset_index()
    
    return df
        
if __name__ == "__main__":
    ER = EvaluateRBP()
    #df = init_df()
    #ticker = "TRIP"
    #tickers_list = ["ADSK", "EBAY", "TRIP", "WDAY"]
    
    #'''
    tickers_list = ["ADP", "ADSK", "AKAM", "ANGI", "ANSS", "AZPN", "BIDU", "CDNS", "CHKP", "CSGP",
    "CTSH", "DOX", "EA", "EBAY", "EXAS", "GRPN", "HCP", "INCY", "INTU", "MANH",
    "MELI", "NICE", "NTES", "PAYX", "PEGA", "SNPS", "SPLK", "SSNC", "TCOM", "TRIP",
    "TTEK", "TTWO", "VRSK", "VRSN", "WDAY"
    ]
    #'''
    
    days_into_future = 10
    scale_market_cap='5 - Large'
    
    df = ED().extract_data_for_prediction_by_group(days_into_future=days_into_future, fama_industry="Business Services", scale_market_cap=scale_market_cap, start_date=datetime.date(2014,11,13), end_date=datetime.date(2023,9,29), additional_features=True, sentiment_feature=True, intraday_actual_vol=True)
    df["iv_error"] = df[f"iv{days_into_future}d"] - df[f"{days_into_future}_days_future_volatility"]
    print(df)
    
    response_column="iv_error"
    predictor_columns = ['iv10d', 'iv20d', 'iv30d', 'iv60d', 'iv90d', 'iv6m', 'volume', 'evebitda','m.marketcap', 'm.pb', 'm.pe']    #, 'avg_daily_sentiment']
    folder_path = "relevance_based_prediction/analysis/business_services_large_cap_w_sentiment_10d"
    #predictor_columns = df.drop(columns=["date", "ticker", "lastupdated", "ev","evebit","evebitda","marketcap", "open", "high", "low", "closeadj", "closeunadj", "10 days future volatility", "20 days future volatility", "30 days future volatility", "DJIA", "SP500"]).columns.to_list()
    #predictor_columns = ['iv10d', 'iv20d', 'iv30d', 'iv60d', 'iv90d', 'iv6m', 'volume', 'evebitda','m.marketcap', 'm.pb', 'm.pe']
    #predicted_vs_actual_df = ER.get_predicted_vs_actual_df(df=df, ticker=ticker, ticker_column="ticker", predictor_columns=predictor_columns, date_column="date", testing_size=0.25, days_into_future=days_into_future, max_lookback=pd.Timedelta(days=120), model="RBP", response_column=response_column, normal_experiment=True, plot_iv_errors_fig=True, r_threshold=0, folder_path=folder_path)
    #print(predicted_vs_actual_df)
    #rbp_rmse, iv_rmse, predicted_corr, pred_win_rate_over_iv, iv_adj_error_pred_df = ER.test(ticker=ticker, predicted_vs_actual_df=predicted_vs_actual_df, days_into_future=days_into_future, model="RBP", mode="iv_errors", folder_path=folder_path)
    ER.batch_test(df=df, tickers_list=tickers_list, ticker_column="ticker", predictor_columns=predictor_columns, date_column="date", testing_size=0.25, days_into_future=days_into_future, max_lookback=pd.Timedelta(days=120), model="RBP", mode="iv_errors", response_column=response_column, normal_experiment=True, r_threshold=0, plot_iv_figures=True, folder_path=folder_path)