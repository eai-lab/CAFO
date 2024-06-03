import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error


def calculate_metrics_by_exp(label_df, y_label_name, y_pred_name):
    """"""

    test_results = dict()
    for exp in list(label_df.EXP.unique()):
        target = label_df.loc[label_df["EXP"] == exp, y_label_name]
        pred = label_df.loc[label_df["EXP"] == exp, y_pred_name]
        mae = mean_absolute_error(target, pred)
        mse = mean_squared_error(target, pred)
        rmse = mean_squared_error(target, pred, squared=False)
        r2 = r2_score(target, pred)
        mape = mean_absolute_percentage_error(target, pred)
        test_results[exp] = [mae, mse, rmse, r2, mape]

    test_results_file = pd.DataFrame(test_results, index=["MAE", "MSE", "RMSE", "R2", "MAPE"]).T
    return test_results_file
