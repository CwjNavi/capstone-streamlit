# need to run with python -m
AUTHORS = ["Yu Liang"]
VERSION = ["v1.0"]

import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

import relevance_based_prediction.evaluate_rbp as EvRBP
from extract_data import ExtractData as ED


class Outlier_Stats_Analysis:
    def __init__(self):
        pass
    
    def compile_test_results_by_ticker(self, days_into_future, tickers):
        test_dict_list = []
        iv_error_dict_list = []
        for ticker in tickers:
            df = ED().extract_data_for_prediction_by_ticker(ticker=ticker, days_into_future=days_into_future)
            predicted_vs_actual_df = self.get_predictions_and_error(df, days_into_future=days_into_future, ticker=ticker, rbp_prediction=False)
        
            df_main_group, df_outlier_group = self.split_outlier_and_non_outlier_group(predicted_vs_actual_df=predicted_vs_actual_df, days_into_future=60)

            iv_error_dict = self.outlier_test_iv_error(df_main_group=df_main_group, df_outlier_group=df_outlier_group)
            iv_error_dict["ticker"] = ticker
            #test_dict_list.append(rbp_iv_error_dict)
            iv_error_dict_list.append(iv_error_dict)
        #pd.DataFrame(test_dict_list).to_csv("relevance_based_prediction/analysis/rbp-iv-stats.csv")
        pd.DataFrame(iv_error_dict_list).to_csv("relevance_based_prediction/analysis/iv_error_stats_by_ticker.csv", index=False)
    
    def compile_test_results_by_industry(self, days_into_future:int, sic_sectors:list[str], by_fama_industry:bool=True, by_dates:list[datetime.date]=[datetime.date(2010,1,1), datetime.date(2018,12,31)], save_folder="relevance_based_prediction/analysis/services"):
        test_dict_list = []
        iv_error_dict_list = []
        for i in range(len(by_dates)-1):
            if i == 0:
                start_date = by_dates[i]
            else:
                start_date = by_dates[i]+ datetime.timedelta(days=1)
            end_date = by_dates[i+1]
            date_range = f"{start_date} to {end_date}"
                        
            for sic_sector in sic_sectors:
                df = ED().extract_data_for_prediction_by_group(days_into_future=days_into_future, sic_sector=sic_sector, start_date=start_date, end_date=end_date)
                
                if by_fama_industry == False:
                    iv_error_dict = self.run_test(df=df, days_into_future=days_into_future, save_folder=save_folder, date_range=date_range)
                    
                    iv_error_dict["sic_sector"] = sic_sector
                    #test_dict_list.append(rbp_iv_error_dict)
                    iv_error_dict_list.append(iv_error_dict)
                    
                else:
                    fama_industries = df["famaindustry"].unique()
                    print(fama_industries)
                    for fama_industry in fama_industries:
                        fama_industry_df = df[df["famaindustry"]==fama_industry]
                        print(len(fama_industry_df), fama_industry)
                        iv_error_dict = self.run_test(df=fama_industry_df, days_into_future=days_into_future, industry=fama_industry, save_folder=save_folder, date_range=date_range)

                        iv_error_dict["date_range"] = f"{start_date} to {end_date}"
                        iv_error_dict["sic_sector"] = sic_sector
                        iv_error_dict["fama_industry"] = fama_industry
                        #test_dict_list.append(rbp_iv_error_dict)
                        iv_error_dict_list.append(iv_error_dict)
            
        #pd.DataFrame(test_dict_list).to_csv("relevance_based_prediction/analysis/rbp-iv-stats.csv")
        if by_fama_industry == True:
            pd.DataFrame(iv_error_dict_list).replace("", "No Data").set_index(["sic_sector", "fama_industry", "date_range"]).to_excel(f"{save_folder}/iv_error_stats_by_industry.xlsx")
        else:
            pd.DataFrame(iv_error_dict_list).replace("", "No Data").set_index(["sic_sector", "date_range"]).to_excel(f"{save_folder}/iv_error_stats_by_sector.xlsx")
        
    def run_test(self, df, days_into_future, industry:Optional[str]=None, save_folder="relevance_based_prediction/analysis/services", date_range:str=""):
        predicted_vs_actual_df = self.get_predictions_and_error(df, days_into_future=days_into_future, rbp_prediction=False)
        
        df_main_group, df_outlier_groups, thresholds, outlier_group_names = self.split_outlier_and_non_outlier_group(predicted_vs_actual_df=predicted_vs_actual_df, days_into_future=60)
        
        self.plot_main_and_outlier_distributions(df_main_group=df_main_group, df_outlier_groups=df_outlier_groups, thresholds=thresholds, outlier_group_names=outlier_group_names, industry=industry, days_into_future=days_into_future, save_folder=save_folder, date_range=date_range)

        iv_error_dict = self.outlier_test_iv_error(df_main_group=df_main_group, df_outlier_groups=df_outlier_groups, thresholds=thresholds, outlier_group_names=outlier_group_names)
        iv_error_dict["date_range"] = date_range
        
        return iv_error_dict
    
    def outlier_test_iv_error(self, df_main_group:pd.DataFrame, df_outlier_groups:list[pd.DataFrame], thresholds:list[tuple], outlier_group_names:list[str]):
        iv_error_dict = {"Main group mean":df_main_group["iv_error"].mean(), "Main group std":df_main_group["iv_error"].std(), "Main group size":len(df_main_group)}
        
        for i, df_outlier_group in enumerate(df_outlier_groups):
            outlier_group_name = outlier_group_names[i]
            outlier_lower_threshold, outlier_upper_threshold = thresholds[i]
            t_stat, p_value = stats.ttest_ind(df_outlier_group["iv_error"], df_main_group["iv_error"], equal_var=False)
            
            iv_error_dict[f"{outlier_group_name} mean"] = df_outlier_group["iv_error"].mean()
            iv_error_dict[f"{outlier_group_name} std"] = df_outlier_group["iv_error"].std()
            iv_error_dict[f"{outlier_group_name} size"] = len(df_outlier_group)
            iv_error_dict[f"{outlier_group_name} threshold"] = outlier_lower_threshold
            iv_error_dict[f"{outlier_group_name} threshold"] = outlier_upper_threshold
            iv_error_dict[f"{outlier_group_name} p_value"] = p_value
        
        return iv_error_dict
    
    def outlier_test_rbp_iv_error(self, df_main_group, df_outlier_group):
        t_stat, p_value = stats.ttest_ind(df_outlier_group["rbp-iv abs error"], df_main_group["rbp-iv abs error"], equal_var=False)
        print("Outlier group size:", len(df_outlier_group))
        print("Main group mean:", df_main_group["rbp-iv abs error"].mean(), "Outlier group mean", df_outlier_group["rbp-iv abs error"].mean())
        print("RBP vs IV test", t_stat, p_value)
        
        rbp_iv_error_dict = {"Main group mean":df_main_group["rbp-iv abs error"].mean(), "Outlier group mean":df_outlier_group["rbp-iv abs error"].mean(), "Outlier group size":len(df_outlier_group), "p_value":p_value}

        return rbp_iv_error_dict
    
    def get_predictions_and_error(self, df, days_into_future, rbp_prediction=False, ticker:Optional[str]=None):
        ER = EvRBP.EvaluateRBP()

        predictor_columns = ["iv10d", "iv20d", "iv30d", "IV{days_into_future}d", "iv90d", "iv6m", "iv1yr"]
        
        predicted_vs_actual_df = df.copy()
        predicted_vs_actual_df["iv_error"] = predicted_vs_actual_df[f"iv{days_into_future}d"]-predicted_vs_actual_df[f"{days_into_future}_days_future_volatility"]
        
        if rbp_prediction == True:
            predicted_vs_actual_df = ER.get_predicted_vs_actual_df(df, predictor_columns=predictor_columns, date_column="date", testing_size=0.25, days_into_future=60, model="RBP")
            ER.test(ticker=ticker, predicted_vs_actual_df=predicted_vs_actual_df, days_into_future=60, model="RBP")
            predicted_vs_actual_df["abs_rbp_error"] = abs(predicted_vs_actual_df["prediction"]-predicted_vs_actual_df[f"{days_into_future}_days_future_volatility"])
        
            predicted_vs_actual_df["rbp-iv abs error"] = predicted_vs_actual_df["abs_rbp_error"] - predicted_vs_actual_df["iv_error"]
        
        return predicted_vs_actual_df
        
    def split_outlier_and_non_outlier_group(self, predicted_vs_actual_df:pd.DataFrame, days_into_future:int) -> tuple[pd.DataFrame, list[pd.DataFrame], list[tuple[float, float]], list[str]]:
        # split group into outlier and normal group
        #iv_std = predicted_vs_actual_df[f"iv{days_into_future}d"].std()
        #iv_mean = predicted_vs_actual_df[f"iv{days_into_future}d"].mean()
        outlier_upper_threshold = predicted_vs_actual_df[f"iv{days_into_future}d"].mean() + 2*predicted_vs_actual_df[f"iv{days_into_future}d"].std()
        outlier_lower_threshold = predicted_vs_actual_df[f"iv{days_into_future}d"].mean() - 2*predicted_vs_actual_df[f"iv{days_into_future}d"].std()
        
        df_main_group = predicted_vs_actual_df[(predicted_vs_actual_df[f"iv{days_into_future}d"] > outlier_lower_threshold) & (predicted_vs_actual_df[f"iv{days_into_future}d"] < outlier_upper_threshold)]
        df_outlier_group = predicted_vs_actual_df[(predicted_vs_actual_df[f"iv{days_into_future}d"] < outlier_lower_threshold) | (predicted_vs_actual_df[f"iv{days_into_future}d"] > outlier_upper_threshold)]
        
        df_main_group = df_main_group.dropna(subset=["iv_error"])
        df_outlier_group = df_outlier_group.dropna(subset=["iv_error"])
        
        return df_main_group, [df_outlier_group], [(outlier_lower_threshold, outlier_upper_threshold)], ["Outlier group"]
    
    def plot_main_and_outlier_distributions(self, df_main_group:pd.DataFrame, df_outlier_groups:list[pd.DataFrame], thresholds:list[tuple], outlier_group_names:list[str], industry, days_into_future, save_folder="relevance_based_prediction/analysis/services", date_range:str="", plot_vol=True):
        def plot_histogram_percentile(data, xlabel, title, save_path):
            fig, ax1 = plt.subplots()
            
            counts, bins = np.histogram(data, bins=30)  # Compute histogram counts and bin edges
            percentages = (counts / counts.sum()) * 100  # Convert frequencies to percentages
            
            plt.bar(bins[:-1], percentages, width=np.diff(bins), alpha=0.5, color='blue', edgecolor='black')
            plt.xlabel(xlabel)
            plt.ylabel('Percentage (%)')  # Changed from 'Frequency' to 'Percentage'
            plt.title(title)
            plt.ylim(0, 60)
            
            fig.subplots_adjust(bottom=0.20)
            
            # Compute mean and standard deviation
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            # Prepare text for the box
            graph_text = f"Mean: {mean_val:.2f}\nStd Dev: {std_val:.2f}\nSize: {len(data)}"
            
            # Add text with a transparent background below the graph
            fig.text(
                0.5, 0.02,  # (x, y) in figure coordinates (0=bottom, 1=top)
                graph_text,
                ha="center", fontsize=8,
                bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5')  # Transparent box
            )
            
            plt.savefig(save_path)
            plt.close()

        # Plot for main group IV errors
        plot_histogram_percentile(
            df_main_group['iv_error'],
            xlabel=f'IV{days_into_future}d errors',
            title=f'Histogram of IV{days_into_future}d main group ({industry})\n{date_range}',
            save_path=f'{save_folder}/distributions/{industry}_main_iv{days_into_future}d_error_{date_range}.png'
        )

        for i, df_outlier_group in enumerate(df_outlier_groups):
            outlier_group_name = outlier_group_names[i]
            outlier_lower_threshold, outlier_upper_threshold = thresholds[i]
            
            # Plot for outlier group IV errors
            plot_histogram_percentile(
                df_outlier_group['iv_error'], 
                xlabel=f'IV{days_into_future}d errors',
                title=f'Histogram of IV{days_into_future}d errors {outlier_group_name} ({industry})\n{date_range}\nOutlier lower threshold: {outlier_lower_threshold}, Outlier upper threshold: {outlier_upper_threshold}',
                save_path=f'{save_folder}/distributions/{industry}_outlier_iv{days_into_future}d_error_{outlier_group_name}_{date_range}.png'
            )

        if plot_vol == True:
            # Plot for IV values
            plot_histogram_percentile(
                pd.concat([df_main_group[f'iv{days_into_future}d'], df_outlier_group[f'iv{days_into_future}d']]),
                xlabel=f'IV{days_into_future}d',
                title=f'Histogram of IV{days_into_future}d ({industry})\n{date_range}',
                save_path=f'{save_folder}/vol/{industry}_iv{days_into_future}d_{date_range}.png'  # Fixed filename issue
            )
            #
            ## Plot for actual future volatility values
            plot_histogram_percentile(
                pd.concat([df_main_group[f'iv{days_into_future}d'], df_outlier_group[f'iv{days_into_future}d']]),
                xlabel=f'{days_into_future}d future volatility',
                title=f'Histogram of actual {days_into_future}d future volatility ({industry})\n{date_range}',
                save_path=f'{save_folder}/vol/{industry}_future_{days_into_future}d_vol_{date_range}.png'  # Fixed filename issue
            )

class Services_Multiple_Outlier_Groups_Stats_Analysis(Outlier_Stats_Analysis):
    def __init__(self):
        super().__init__
    
    def split_outlier_and_non_outlier_group(self, predicted_vs_actual_df:pd.DataFrame, days_into_future:int) -> tuple[pd.DataFrame, list[pd.DataFrame], list[tuple[float, float]], list[str]]:
        # split group into outlier and normal group
        df_main_group = predicted_vs_actual_df[(predicted_vs_actual_df[f"iv{days_into_future}d"] > 25) & (predicted_vs_actual_df[f"iv{days_into_future}d"] < 50)].dropna(subset=["iv_error"])
        df_outlier_group_1 = predicted_vs_actual_df[(predicted_vs_actual_df[f"iv{days_into_future}d"] < 25)].dropna(subset=["iv_error"])
        df_outlier_group_2 = predicted_vs_actual_df[(predicted_vs_actual_df[f"iv{days_into_future}d"] > 50) & (predicted_vs_actual_df[f"iv{days_into_future}d"] < 75)].dropna(subset=["iv_error"])
        df_outlier_group_3 = predicted_vs_actual_df[(predicted_vs_actual_df[f"iv{days_into_future}d"] > 75) & (predicted_vs_actual_df[f"iv{days_into_future}d"] < 100)].dropna(subset=["iv_error"])
        df_outlier_group_4 = predicted_vs_actual_df[(predicted_vs_actual_df[f"iv{days_into_future}d"] > 100)].dropna(subset=["iv_error"])
        
        return df_main_group, [df_outlier_group_1, df_outlier_group_2, df_outlier_group_3, df_outlier_group_4], [(0, 25), (50, 75), (75, 100), (100, "inf")], ["Outlier group 1", "Outlier group 2", "Outlier group 3", "Outlier group 4"]
    
    # only here to turn off plotting of vol
    def run_test(self, df, days_into_future, industry:Optional[str]=None, save_folder="relevance_based_prediction/analysis/services", date_range:str=""):
        predicted_vs_actual_df = self.get_predictions_and_error(df, days_into_future=days_into_future, rbp_prediction=False)
        
        df_main_group, df_outlier_groups, thresholds, outlier_group_names = self.split_outlier_and_non_outlier_group(predicted_vs_actual_df=predicted_vs_actual_df, days_into_future=60)
        
        self.plot_main_and_outlier_distributions(df_main_group=df_main_group, df_outlier_groups=df_outlier_groups, thresholds=thresholds, outlier_group_names=outlier_group_names, industry=industry, days_into_future=days_into_future, save_folder=save_folder, date_range=date_range, plot_vol=False)

        iv_error_dict = self.outlier_test_iv_error(df_main_group=df_main_group, df_outlier_groups=df_outlier_groups, thresholds=thresholds, outlier_group_names=outlier_group_names)
        iv_error_dict["date_range"] = date_range
        
        return iv_error_dict

if __name__ == "__main__":
    OSA = Outlier_Stats_Analysis()
    #OSA.outlier_test(ticker="AAPL", days_into_future=60)
    #OSA.compile_test_results_by_ticker(days_into_future=60, tickers=["MSFT"])   #, "AMZN", "META", "AAPL", "GOOG", "NVDA", "TSLA"])
    #sic_sectors = list(ED.run_query("select distinct sicsector from ticker_sectors")["sicsector"])
    #print(sic_sectors)
    #OSA.compile_test_results_by_industry(days_into_future=60, sic_sectors=["Services"], by_fama_industry=True, by_dates=[datetime.date(2010,1, 1), datetime.date(2018, 12, 31), datetime.date(2021, 12, 31), datetime.date(2024, 1, 1)], save_folder="relevance_based_prediction/analysis/services")
    
    SMOGSA = Services_Multiple_Outlier_Groups_Stats_Analysis()
    SMOGSA.compile_test_results_by_industry(days_into_future=60, sic_sectors=["Services"], by_fama_industry=True, by_dates=[datetime.date(2010,1, 1), datetime.date(2018, 12, 31), datetime.date(2021, 12, 31), datetime.date(2024, 1, 1)], save_folder="relevance_based_prediction/analysis/services_multiple_outliers")