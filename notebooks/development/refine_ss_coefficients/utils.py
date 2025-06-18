import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def linear_regression_summary(
    data, pred_vars, resp_var, positive=False, fit_intercept=True, plot=False
):
    """
    Fit a linear regression model using scikit-learn and print a summary.

    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset containing the predictor and response variables.
    pred_vars : list of str
        List of column names in `data` to be used as predictor variables.
    resp_var : str
        Name of the response variable column in `data`.
    positive : bool, optional (default=False)
        If True, forces the regression coefficients to be non-negative.
    fit_intercept : bool, optional (default=True)
        If True, includes an intercept in the model.
    plot : bool, optional (default=False)
        If True, displays diagnostic plots including:
            - Residuals vs Fitted
            - Predicted vs Observed
            - Histogram of Residuals
            - Q-Q Plot of Residuals

    Prints:
    -------
    - R² and Adjusted R²
    - AIC (Akaike Information Criterion)
    - Coefficient estimates with:
        - p-values
        - 95% confidence intervals

    Notes:
    ------
    - Confidence intervals are calculated using the t-distribution critical value.
    - Uses pseudo-inverse for numerical stability in variance estimation.
    - Designed to mimic the output of statsmodels.OLS while using scikit-learn for fitting.
    """
    # Define predictors and response
    X = data[pred_vars]
    y = data[resp_var]

    # Fit the model
    model = LinearRegression(positive=positive, fit_intercept=fit_intercept)
    model.fit(X, y)
    y_pred = model.predict(X)

    # Calculate R² and Adjusted R²
    if fit_intercept:
        r2 = r2_score(y, y_pred)
        adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - len(pred_vars) - 1)
    else:
        ss_total_uncentered = np.sum(y**2)
        ss_residual = np.sum((y - y_pred) ** 2)
        r2 = 1 - ss_residual / ss_total_uncentered
        adj_r2 = 1 - (1 - r2) * len(y) / (len(y) - len(pred_vars))

    # Calculate AIC using full log-likelihood
    n = len(y)
    residual_sum_of_squares = np.sum((y - y_pred) ** 2)
    sigma2 = residual_sum_of_squares / n
    log_likelihood = -n / 2 * np.log(2 * np.pi * sigma2) - residual_sum_of_squares / (
        2 * sigma2
    )
    aic = 2 * (len(pred_vars) + int(fit_intercept)) - 2 * log_likelihood

    # Calculate p-values and confidence intervals
    params = np.append(model.intercept_, model.coef_) if fit_intercept else model.coef_
    newX = np.append(np.ones((len(X), 1)), X, axis=1) if fit_intercept else X
    MSE = np.sum((y - y_pred) ** 2) / (len(newX) - newX.shape[1])

    var_b = MSE * (np.linalg.pinv(np.dot(newX.T, newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params / sd_b
    df = len(newX) - newX.shape[1]
    p_values = [2 * (1 - stats.t.cdf(np.abs(i), df)) for i in ts_b]

    # Confidence intervals using t-distribution critical value
    t_crit = stats.t.ppf(1 - 0.025, df)
    conf_ints = [
        [params[i] - t_crit * sd_b[i], params[i] + t_crit * sd_b[i]]
        for i in range(len(params))
    ]

    # Print summary
    print("=== Linear Regression Summary ===")
    print(f"R²:          {r2:.4f}")
    print(f"Adjusted R²: {adj_r2:.4f}")
    print(f"AIC:         {aic:.1f}")
    print("\nCoefficients:")
    if fit_intercept:
        print(
            f"Intercept: {params[0]:.2f} (p-value: {p_values[0]:.4f}, 95% CI: [{conf_ints[0][0]:.2f}, {conf_ints[0][1]:.2f}])"
        )
    for i, coef in enumerate(model.coef_):
        idx = i + int(fit_intercept)
        print(
            f"{pred_vars[i]}: {coef:.2f} (p-value: {p_values[idx]:.4f}, 95% CI: [{conf_ints[idx][0]:.2f}, {conf_ints[idx][1]:.2f}])"
        )

    # Diagnostic plots
    if plot:
        residuals = y - y_pred
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        sns.scatterplot(x=y_pred, y=residuals, ax=axs[0, 0])
        axs[0, 0].axhline(0, color="gray", linestyle="--")
        axs[0, 0].set_title("Residuals vs Fitted")
        axs[0, 0].set_xlabel("Fitted values")
        axs[0, 0].set_ylabel("Residuals")

        sns.scatterplot(x=y, y=y_pred, ax=axs[0, 1])
        axs[0, 1].plot([y.min(), y.max()], [y.min(), y.max()], "r--")
        axs[0, 1].set_title("Predicted vs Observed")
        axs[0, 1].set_xlabel("Observed values")
        axs[0, 1].set_ylabel("Predicted values")

        sns.histplot(residuals, kde=True, ax=axs[1, 0])
        axs[1, 0].set_title("Histogram of Residuals")

        sm.qqplot(residuals, line="s", ax=axs[1, 1])
        axs[1, 1].set_title("Q-Q Plot of Residuals")

        plt.tight_layout()
        plt.show()