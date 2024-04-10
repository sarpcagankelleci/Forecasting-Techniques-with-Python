
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats 
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import norm
from itertools import product
from bbplot import bijan


# # QUESTION 1


df = pd.read_csv('renault.csv', skiprows=0, header=None)
df = df.iloc[:, :2]
df.columns = ['Date', 'Sales']
df['Date'] = pd.to_datetime(df['Date'])
df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')


# # Part A

plt.figure(figsize=(10, 6))  

plt.plot(df['Date'], df['Sales'], linestyle='-')  

plt.title('Monthly Sales')  
plt.xlabel('Date')  
plt.ylabel('Sales') 
plt.xticks(rotation=45)  
plt.tight_layout() 

plt.show()


# #  Part B

# Tau = 1

df_monthly = df.resample('M', on='Date').sum()

for i in range(1, len(df_monthly)):
    df_monthly.loc[df_monthly.index[i], 'Forecast'] = df_monthly.loc[df_monthly.index[i - 1], 'Sales']

df_monthly['Error'] = df_monthly['Sales'] - df_monthly['Forecast']


start_index = df[df['Date'] == '2019-01-01'].index[0]

mae = (df_monthly['Sales'][start_index:] - df_monthly['Forecast'][start_index:]).abs().mean()
mape = ((df_monthly['Sales'][start_index:] - df_monthly['Forecast'][start_index:]) / df_monthly['Sales'][start_index:]).abs().mean() * 100
rmse = ((df_monthly['Sales'][start_index:] - df_monthly['Forecast'][start_index:]) ** 2).mean() ** 0.5

print("Mean Absolute Error (MAE):", mae)
print("Mean Absolute Percentage Error (MAPE):", mape)
print("Root Mean Squared Error (RMSE):", rmse)

plt.figure(figsize=(10, 6))

plt.plot(df_monthly.index, df_monthly['Sales'], label='Actual Sales', linestyle='-')

plt.plot(df_monthly.index[:], df_monthly['Forecast'][:], label='Forecast', linestyle='--')

plt.title('Monthly Sales Forecast')  
plt.xlabel('Date')  
plt.ylabel('Sales') 
plt.xticks(rotation=45)  
plt.legend()
plt.tight_layout() 

plt.show()


# Tau = 12

df_monthly = df.resample('M', on='Date').sum()

for i in range(1, len(df_monthly)):
    df_monthly.loc[df_monthly.index[i], 'Forecast'] = df_monthly.loc[df_monthly.index[i - 12], 'Sales']

df_monthly['Error'] = df_monthly['Sales'] - df_monthly['Forecast']


start_index = df[df['Date'] == '2019-01-01'].index[0]

mae = (df_monthly['Sales'][start_index:] - df_monthly['Forecast'][start_index:]).abs().mean()
mape = ((df_monthly['Sales'][start_index:] - df_monthly['Forecast'][start_index:]) / df_monthly['Sales'][start_index:]).abs().mean() * 100
rmse = ((df_monthly['Sales'][start_index:] - df_monthly['Forecast'][start_index:]) ** 2).mean() ** 0.5

print("Mean Absolute Error (MAE):", mae)
print("Mean Absolute Percentage Error (MAPE):", mape)
print("Root Mean Squared Error (RMSE):", rmse)

plt.figure(figsize=(10, 6))

plt.plot(df_monthly.index, df_monthly['Sales'], label='Actual Sales', linestyle='-')

plt.plot(df_monthly.index[12:], df_monthly['Forecast'][12:], label='Forecast', linestyle='--')

plt.title('Monthly Sales Forecast')  
plt.xlabel('Date')  
plt.ylabel('Sales') 
plt.xticks(rotation=45)  
plt.legend()
plt.tight_layout() 

plt.show()


# # Part C & D

def moving_average_forecast(data, window=3):
    return data.rolling(window=window, min_periods=1).mean().shift(1)


df_monthly['Moving_Average_Forecast'] = moving_average_forecast(df_monthly['Sales'])


df_2019_onwards = df_monthly.loc['2019-01-01':]

def calculate_errors(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    return mae, mape, rmse

mae_ma, mape_ma, rmse_ma = calculate_errors(df_2019_onwards['Sales'], df_2019_onwards['Moving_Average_Forecast'])

print("\nMoving Average Forecast (3-period) Errors:")
print("MAE:", mae_ma)
print("MAPE:", mape_ma)
print("RMSE:", rmse_ma)

rmse_estimate = np.sqrt(mean_squared_error(df_monthly.loc['2014-01-01':'2018-12-01']['Sales'], 
                                             df_monthly.loc['2014-01-01':'2018-12-01']['Moving_Average_Forecast']))

forecast_2022 = df_monthly.loc['2022-01-01': '2022-12-01', 'Moving_Average_Forecast']
lower_bound = forecast_2022 - 1.645 * rmse_estimate
upper_bound = forecast_2022 + 1.645 * rmse_estimate

print("\nPrediction Intervals for 2022:")
print("Date\t\t\tLower Bound\t\t\tUpper Bound")
for date, lower, upper in zip(lower_bound.index, lower_bound, upper_bound):
    print(f"{date}\t{lower}\t{upper}")

plt.plot(df_monthly.index, df_monthly['Moving_Average_Forecast'], linestyle='--', label='Moving Average Forecast (3-period)')

plt.title('Monthly Sales and Forecasts')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

bijan.eplot(df_2019_onwards['Sales']-df_2019_onwards['Moving_Average_Forecast'])


# # PART D

# Read data from CSV file
data = pd.read_csv("renault.csv", usecols=[0, 1], header=None)
data.columns = ['Month', 'Sales']

# Calculate 3-month moving average for forecasting
data['Forecast'] = data['Sales'].rolling(window=3).mean()

# Drop rows with NaN values due to rolling mean calculation
data.dropna(inplace=True)

def res_analysis(months, sales, forecast, plots=True):
    residuals = forecast - sales
    std_res = (residuals - residuals.mean()) / residuals.std()
    
    mean_residuals = residuals.mean()
    mean_sales = sales.mean()
    mean_residuals_percentage = (mean_residuals / mean_sales) * 100
    
    if plots:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))
        axs[0][0].plot(months, residuals)
        axs[0][0].set_xlabel('Month')
        axs[0][0].set_ylabel('Standardized Residuals')
        axs[0][0].grid(True)
        axs[0][0].hlines(0, min(months), max(months), 'r')

        kde = sm.nonparametric.KDEUnivariate(residuals)
        kde.fit()  # Estimate the densities
        axs[1][0].hist(residuals)
        ax_kde = axs[1][0].twinx()
        ax_kde.plot(kde.support, kde.density, 'r')

        sm.qqplot(std_res, ax=axs[0][1])
        sm.qqline(axs[0][1], "45")
        sm.graphics.tsa.plot_acf(std_res, ax=axs[1][1])

    print(f'Mean of Residual: {mean_residuals:.4f}')
    print(f'S.D. of Residual: {residuals.std():.4f}')
    print(f'MAE : {abs(residuals).mean():.4f}')
    print(f'MAPE: {(abs(residuals) / sales * 100).mean():.4f}')
    print(f'MSE : {sm.tools.eval_measures.mse(sales, forecast):.4f}')
    print(f'RMSE: {sm.tools.eval_measures.rmse(sales, forecast):.4f}')
    print(f'Mean of Residual / Mean of Sales: {mean_residuals_percentage:.2f}%')
    
    return residuals, std_res

# Prepare data for analysis
months_a = data['Month'][3:].reset_index(drop=True)
sales_a = data['Sales'][3:].reset_index(drop=True)
forecast_a = data['Forecast'][:-3].reset_index(drop=True) 

# Plot actual sales and forecasted values
plt.figure(figsize=(14, 4))
plt.plot(months_a, sales_a)
plt.plot(months_a, forecast_a)
plt.xlabel('Months')
plt.legend(['Sales', 'Forecast'])
plt.show()

# Analyze residuals
_ = res_analysis(months_a, sales_a, forecast_a, plots=True)


# # PART E

def exponential_smoothing(data, alpha, start_index=72):
    forecast = [np.nan] * len(data)
    forecast[0:start_index] = data['Sales'].iloc[0:start_index]
    for t in range(start_index, len(data)):
        if t == start_index:
            forecast[t] = np.mean(forecast[0:start_index])
        forecast[t] = alpha * data['Sales'][t - 1] + (1 - alpha) * forecast[t - 1]
    return forecast

df = pd.read_csv('renault.csv', skiprows=0, header=None)
df = df.iloc[:, :2]
df.columns = ['Date', 'Sales']
df['Date'] = pd.to_datetime(df['Date'])
df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')

df_monthly = df.resample('M', on='Date').sum()
alphas = np.linspace(0.1, 1, 10)  
maes = []

for alpha in alphas:
    forecast = exponential_smoothing(df_monthly, alpha)
    mae = mean_absolute_error(df_monthly['Sales'], forecast)
    maes.append(mae)

best_alpha = alphas[np.argmin(maes)]
print("Best alpha:", best_alpha)

forecast_best = exponential_smoothing(df_monthly, best_alpha)

forecast_years = df_monthly.loc['2019-01-01':'2022-12-01']
forecast_best_years = forecast_best[72:-1]

mae = mean_absolute_error(forecast_years['Sales'], forecast_best_years)
mape = np.mean(np.abs((forecast_years['Sales'] - forecast_best_years) / forecast_years['Sales'])) * 100
rmse = np.sqrt(mean_squared_error(forecast_years['Sales'], forecast_best_years))

print("MAE:", mae)
print("MAPE:", mape)
print("RMSE:", rmse)

plt.figure(figsize=(10, 6))
plt.plot(df_monthly.index, df_monthly['Sales'], linestyle='-', label='Actual Sales')
plt.plot(df_monthly.index[72:], forecast_best[72:], label='Forecast', linestyle='--')
plt.title('Exponential Smoothing Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# # PART F

# Function to read data from a CSV file
def read_data(file_path):
    return pd.read_csv(file_path, header=None)

# Function to preprocess data and extract demand column
def preprocess_data(data):
    df = pd.DataFrame()
    df['Date'] = data.loc[:, 0]
    df['Year'] = data.loc[:, 0].apply(lambda x: int(x.split('-')[0]))
    df['Demand'] = data.loc[:, 1]
    return df['Demand']

# Function to get the end period index given a year and quarter
def get_end(year, q=1):
    if q == 1:
        return (year - 2013) * 12 + 11
    return (year - 2010) * 12 + 11

# Function to get the start period index given a year and quarter
def get_start(year, q=1):
    if q == 1:
        return (year - 2013) * 12
    return (year - 2010) * 12

# Function to calculate absolute error between demand and forecast
def get_error(demand, forecast, period, start_year, end_year, q=1):
    if forecast[period] is None or np.isnan(forecast[period]):
        return None
    if period < get_start(start_year, q) or period > get_end(end_year, q):
        return None
    return abs(demand.loc[period] - forecast[period])

# Function to calculate Mean Absolute Percentage Error (MAPE)
def calculate_mape(demand, forecast, start_year, end_year, q=1):
    total_error = 0
    count = 0
    for period in range(len(demand)):
        error = get_error(demand, forecast, period, start_year, end_year, q)
        if error is None:
            continue
        count += 1
        total_error += abs(error / demand.loc[period])
    return 100 * total_error / count

# Function to calculate Mean Absolute Error (MAE)
def calculate_mae(demand, forecast, start_year, end_year, q=1):
    total_error = 0
    count = 0
    for period in range(len(demand)):
        error = get_error(demand, forecast, period, start_year, end_year, q)
        if error is None:
            continue
        count += 1
        total_error += error
    return total_error / count

# Function to calculate Root Mean Square Error (RMSE)
def calculate_rmse(demand, forecast, start_year, end_year):
    total_squared_error = 0
    count = 0
    for period in range(get_start(start_year), get_end(end_year) + 1):
        error = get_error(demand, forecast, period, start_year, end_year)
        if error is not None:
            total_squared_error += error ** 2
            count += 1
    mean_squared_error = total_squared_error / count
    return np.sqrt(mean_squared_error)

# Function to perform double exponential smoothing forecast
def double_exp_forecast(demand, alpha, beta, start, level, slope, k=1):
    l = level
    s = slope
    forecasts = []
    for i in range(k):
        forecasts.append(None)
    for period in range(len(demand)):
        if period < start - k + 1:
            forecasts.append(None)
            continue
        lt_1 = alpha * demand.loc[period - 1] + (1 - alpha) * (l + s)
        st_1 = beta * (lt_1 - l) + (1 - beta) * s
        forecast = lt_1 + k * st_1
        forecasts.append(forecast)
        l = lt_1
        s = st_1
    return forecasts

# Function to initialize level and slope for double exponential smoothing
def initialize(demand, start, k=1):
    rd = demand.loc[:start + k - 1]
    X = np.array(range(len(rd)))
    X = sm.add_constant(X)
    y = rd
    m = sm.OLS(y, X)
    r = m.fit()
    slope = r.params[1]
    level = r.params[0]
    return level, slope

# Function to calculate prediction intervals
def get_interval(forecast, confidence_level, rmse):
    alpha = 1 - confidence_level
    z_value = norm.ppf(1 - alpha / 2)
    std = rmse
    offset = std * z_value
    return (forecast - offset, forecast + offset)

# Function to generate a list of prediction intervals
def get_interval_list(forecast, confidence_level, rmse, start_year=2022, q_num=1):
    forecast = forecast[get_start(start_year, q_num): get_end(start_year, q_num) + 1]
    return [get_interval(f, confidence_level, rmse) for f in forecast]

# Main part of the script
if __name__ == "__main__":
    # Load data
    file_path = "renault.csv"
    data_renault = read_data(file_path)
    demand_renault = preprocess_data(data_renault)

    # Initialization
    start_period_renault = get_start(2019)
    initial_level_renault, initial_slope_renault = initialize(demand_renault, start_period_renault)
    best_mape_renault = float('inf')
    best_mae_renault = float('inf')
    best_rmse_renault = float('inf')
    best_alpha_mape_renault = None
    best_beta_mape_renault = None
    best_alpha_mae_renault = None
    best_beta_mae_renault = None
    best_alpha_rmse_renault = None
    best_beta_rmse_renault = None

    # Search for optimal alpha and beta
    for alpha_renault in np.arange(0.1, 1.1, 0.1):
        for beta_renault in np.arange(0.1, 1.1, 0.1):
            forecast_renault = double_exp_forecast(demand_renault, alpha_renault, beta_renault, start_period_renault, initial_level_renault, initial_slope_renault)

            mape_renault = calculate_mape(demand_renault, forecast_renault, 2019, 2022)
            mae_renault = calculate_mae(demand_renault, forecast_renault, 2019, 2022)
            rmse_renault  = calculate_rmse(demand_renault, forecast_renault, 2019, 2022)

            if mape_renault < best_mape_renault:
                best_mape_renault = mape_renault
                best_alpha_mape_renault = round(alpha_renault, 1)
                best_beta_mape_renault = round(beta_renault, 1)
            if mae_renault < best_mae_renault:
                best_mae_renault = mae_renault
                best_alpha_mae_renault = round(alpha_renault, 1)
                best_beta_mae_renault = round(beta_renault, 1)
            if rmse_renault < best_rmse_renault:
                best_rmse_renault = rmse_renault
                best_alpha_rmse_renault = round(alpha_renault, 1)
                best_beta_rmse_renault = round(beta_renault, 1)

    # Print results
    print("Best MAPE (Renault):", best_mape_renault)
    print("Alpha (MAPE):", best_alpha_mape_renault)
    print("Beta (MAPE):", best_beta_mape_renault)

    print("Best RMSE (Renault):", best_rmse_renault)
    print("Alpha (RMSE):", best_alpha_rmse_renault)
    print("Beta (RMSE):", best_beta_rmse_renault)

    print("Best MAE (Renault):", best_mae_renault)
    print("Alpha (MAE):", best_alpha_mae_renault)
    print("Beta (MAE):", best_beta_mae_renault)

    # Generate optimal forecasts
    optimal_forecast_mape_renault = double_exp_forecast(demand_renault, best_alpha_mape_renault, best_beta_mape_renault, start_period_renault, initial_level_renault, initial_slope_renault)
    optimal_forecast_mae_renault = double_exp_forecast(demand_renault, best_alpha_mae_renault, best_beta_mae_renault, start_period_renault, initial_level_renault, initial_slope_renault)
    optimal_forecast_rmse_renault = double_exp_forecast(demand_renault, best_alpha_rmse_renault, best_beta_rmse_renault, start_period_renault, initial_level_renault, initial_slope_renault)

    # Plot forecasts for Renault
    plt.figure(figsize=(10, 5), facecolor='lightgrey')
    plt.xticks(range(0, len(demand_renault), 12), demand_renault.index[::12], rotation=45)
    plt.title(f'Forecast Analysis for Renault')
    plt.xlabel('Time Basis')
    plt.ylabel('Demand Amount')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Plot actual demand for Renault
    plt.plot(demand_renault, color="black", label='Actual Demand (Renault)')

    # Plot forecasts for Renault with different error metrics
    plt.plot(optimal_forecast_mape_renault, color='blue', linestyle='-', label=f'Forecast (MAPE)\nα* = {best_alpha_mape_renault}, β* = {best_beta_mape_renault}')
    plt.plot(optimal_forecast_mae_renault, color='red', linestyle='-', label=f'Forecast (MAE)\nα* = {best_alpha_mae_renault}, β* = {best_beta_mae_renault}')
    plt.plot(optimal_forecast_rmse_renault, color='green', linestyle='-', label=f'Forecast (RMSE)\nα* = {best_alpha_rmse_renault}, β* = {best_beta_rmse_renault}')

    plt.legend()
    plt.show()

    # Calculate prediction intervals for Renault
    alpha_star_renault = best_alpha_rmse_renault
    beta_star_renault = best_beta_rmse_renault
    std_estimator_renault = calculate_rmse(demand_renault, optimal_forecast_rmse_renault, 2019, 2022)
    prediction_intervals_renault = get_interval_list(optimal_forecast_rmse_renault, 0.9, std_estimator_renault)

    # Extract lower and upper bounds for Renault
    lower_bound_renault = [interval[0] for interval in prediction_intervals_renault]
    upper_bound_renault = [interval[1] for interval in prediction_intervals_renault]

    # Print prediction intervals for Renault along with the corresponding forecasts
    print(f"\n90% Confidence Prediction Intervals for Renault:")
    for interval, forecast in zip(prediction_intervals_renault, optimal_forecast_rmse_renault):
        print(f"Forecast: {forecast}, Interval: {interval}")

    
    # Plot demand over time with prediction intervals for Renault
    plt.figure(figsize=(10, 5), facecolor='lightgrey')
    plt.plot(demand_renault, color="black", label='Demand Over Time')
    plt.fill_between(range(9 * 12, len(lower_bound_renault) + 9 * 12 ), lower_bound_renault, upper_bound_renault, color='red', alpha=0.3, label='Prediction Intervals')
    plt.xticks(range(0, len(demand_renault), 12), range(2013, 2013 + len(demand_renault) // 12), rotation=45)
    plt.title(f'Renault Demand Over Time - 90% Prediction Intervals')
    plt.xlabel('Year Basis')
    plt.ylabel('Demand Over Time')
    plt.legend()
    plt.tight_layout()
    plt.show()


# # PART G

# To evaluate the residual diagnostics for independence and normality, we need to analyze the behavior of the residuals obtained from the double exponential smoothing forecast.
# 
# Independence of Residuals: We can examine whether the residuals exhibit any patterns or trends over time. If there is any systematic pattern in the residuals, it suggests that the model might not have captured all the underlying structures in the data, leading to biased forecasts. We can plot the residuals against time or against forecasted values to detect any systematic patterns.
# 
# Normality of Residuals: To assess the normality of residuals, we can create a histogram of the residuals and compare it to a normal distribution. Additionally, statistical tests such as the Shapiro-Wilk test can be performed to formally test for normality.
# 
# Regarding the comparison to previous forecasts, we can observe how the new forecasts based on the optimal alpha and beta values perform compared to the previous forecasts. We should compare the accuracy metrics (MAPE, RMSE, MAE) of the new forecasts with those of the previous forecasts. If the new forecasts have lower error metrics, it indicates that they are more accurate and provide better predictions.
# 
# A drawback of the double exponential smoothing forecast with respect to this data is that it might not capture more complex patterns or seasonality present in the demand data. Double exponential smoothing is a simple forecasting method that assumes a linear trend, which might not be suitable for data with more complex patterns. If the demand data exhibits strong seasonality or non-linear trends, a more sophisticated forecasting method such as seasonal ARIMA or dynamic regression might provide better forecasts.
# 
# In summary, to assess the quality of the double exponential smoothing forecast and compare it to previous forecasts, we need to analyze the residuals for independence and normality, compare accuracy metrics, and consider whether the model captures the underlying patterns in the data effectively. Additionally, understanding the limitations of the forecasting method in capturing complex patterns is essential for making informed decisions about model selection.
# 
# 
# 
# 
# 



# Function to read data from a CSV file
def read_data(file_path):
    return pd.read_csv(file_path, header=None)

# Function to preprocess data and extract demand column
def preprocess_data(data):
    df = pd.DataFrame()
    df['Date'] = data.loc[:, 0]
    df['Year'] = data.loc[:, 0].apply(lambda x: int(x.split('-')[0]))
    df['Demand'] = data.loc[:, 1]
    return df['Demand']

def get_end_time(year, question = 1):
    if question == 1:
        return (year - 2013) * 12 + 11
    return (year - 2010) * 12 + 11

def get_start_time(year, question = 1):
    if question == 1:
        return (year - 2013) * 12
    return (year - 2010) * 12

def is_valid_time(start_year, end_year, t, question = 1):
    return t < get_start_time(start_year, question) or get_end_time(end_year, question) < t

def get_residuals(demand, forecast, start_year, end_year, question=1):
    out = []
    for t in range(len(demand)):
        if forecast[t] is None or np.isnan(forecast[t]):  # Check if forecast value is None or NaN
            continue
        if is_valid_time(start_year, end_year, t, question):
            continue
        out.append(demand.loc[t] - forecast[t])
    return out

# Residual diagnostics
def residual_diagnostics(residuals):
    # Independence check (ACF plot)
    plt.figure(figsize=(10, 5))
    plot_acf(residuals, lags=20, ax=plt.gca(), title='Autocorrelation of Residuals')
    plt.show()

    # Normality check (Q-Q plot)
    plt.figure(figsize=(10, 5))
    probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.show()

# Compare to previous forecasts
def compare_forecasts(actual_demand, forecasts):
    plt.figure(figsize=(10, 5))
    plt.plot(actual_demand, label="Actual Demand", color='black')
    for i, forecast in enumerate(forecasts, 1):
        plt.plot(forecast, label=f"Forecast {i}")
    plt.title("Comparison of Forecasts with Actual Demand")
    plt.xlabel("Time")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(True)
    plt.show()

# Assess drawback of the forecast
def assess_drawback(residuals):
    plt.figure(figsize=(10, 5))
    plt.plot(residuals)
    plt.title("Residual Plot")
    plt.xlabel("Time")
    plt.ylabel("Residual")
    plt.grid(True)
    plt.show()

    # You can also calculate and print summary statistics of residuals
    print("Summary Statistics of Residuals:")
    print(pd.Series(residuals).describe())

if __name__ == "__main__":
    # Load data
    file_path = "renault.csv"
    data_renault = read_data(file_path)
    demand_renault = preprocess_data(data_renault)

    # Compute residuals
    residuals = get_residuals(demand_renault, optimal_forecast_rmse_renault, 2019, 2022)

    # Perform residual diagnostics
    residual_diagnostics(residuals)

    # Compare to previous forecasts
    compare_forecasts(demand_renault, [optimal_forecast_mape_renault, optimal_forecast_mae_renault, optimal_forecast_rmse_renault])

    # Assess drawback of the forecast
    assess_drawback(residuals)

# Calculate residuals
residuals = calculate_residuals(demand, optimal_forecast_renault, 2019, 2022)
print("Mean Residuals:", np.mean(residuals))


# # PART H

# Initialization for double exponential smoothing
start_year = 2022
start_time = get_start(start_year)

def initialize(demand, start, k=1):
    rd = demand.loc[:start + k - 1]
    X = np.array(range(len(rd)))
    X = sm.add_constant(X)
    y = rd
    m = sm.OLS(y, X)
    r = m.fit()
    slope = r.params[1]
    level = r.params[0]
    return level, slope

def get_error(demand, forecast, period, start_year, end_year, q=1):
    start_period = get_start(start_year, q)
    end_period = min(get_end(end_year, q), len(demand) - 1)
    
    if period < start_period or period > end_period:
        return None
    
    demand_index = period - q
    if demand_index < 0 or demand_index >= len(demand):
        return None
    
    demand_value = demand.iloc[demand_index]
    forecast_value = forecast[demand_index]
    
    if demand_value is None or forecast_value is None:
        return None
    
    return abs(demand_value - forecast_value)

reg_level, reg_slope = initialize(demand_renault, start_time)

# Forecasting for three and six months ahead
three_month_ahead_forecast = double_exp_forecast(demand_renault, alpha_star_renault, beta_star_renault, start_time, reg_level, reg_slope, 3)
six_month_ahead_forecast = double_exp_forecast(demand_renault, alpha_star_renault, beta_star_renault, start_time, reg_level, reg_slope, 6)

# Plotting forecasts
plt.figure(figsize=(10, 5), facecolor='lightgrey')
plt.xticks(range(0, len(demand_renault), 12), demand_renault.index[::12], rotation=45)
plt.title(f'Forecast Graph Over Time - DE Smoothing')
plt.xlabel('Year Basis')
plt.ylabel('Demand')
plt.tight_layout()
plt.plot(demand_renault, color="black", label='Actual Demand (Renault)')
plt.plot(three_month_ahead_forecast, color='red', label='DE smoothing 3 month ahead')
plt.plot(six_month_ahead_forecast, color='blue', label='DE smoothing 6 month ahead')
plt.legend()
plt.show()

# Forecasting evaluation
for k in (3, 6):
    forecast = double_exp_forecast(demand_renault, alpha_star_renault, beta_star_renault, get_start(2019), reg_level, reg_slope, k)
    mape = calculate_mape(demand_renault, forecast, 2019, 2022)
    mae = calculate_mae(demand_renault, forecast, 2019, 2022)
    rmse = calculate_rmse(demand_renault, forecast, 2019, 2022)
    
    print(f"Performance Metrics for {k} Month Lookahead Forecast")
    print("Mean Absolute Percentage Error (MAPE):", mape)
    print("Mean Absolute Error (MAE):", mae)
    print("Root Mean Square Error (RMSE):", rmse)
    print("Updated Alpha:", alpha_star_renault)
    print("Updated Beta:", beta_star_renault)


# # PART I


# Function to transform demand data to remove seasonality
def transform_demand_to_U(demand_data):
    transformed_data = []
    for t in range(len(demand_data)):
        if t < 12:
            transformed_data.append(None)
        else:
            transformed_data.append(demand_data.loc[t] - demand_data.loc[t - 12])
    return pd.DataFrame(transformed_data).loc[:,0]

# Apply transformation to demand data
U_style = transform_demand_to_U(demand_renault)

# Plot the transformed series
plt.figure(figsize=(10, 5))
plt.plot(U_style)
plt.xticks(range(0, len(df['Year']), 12), df['Year'].unique(), rotation=45)
plt.title('Transformed Demand Over Time (2013-2022)')
plt.xlabel('Year')
plt.ylabel('Demand')
plt.tight_layout()
plt.show()



# Define the functions for error calculation
def calculate_mean_abs_error(demand_data, forecast_data, start_yr, end_yr, question=1):
    total_error = 0
    data_count = 0
    for t_idx in range(len(demand_data)):
        if forecast_data[t_idx] is None or np.isnan(forecast_data[t_idx]):
            continue
        if check_valid_time(start_yr, end_yr, t_idx, question):
            continue
        error = abs(demand_data[t_idx] - forecast_data[t_idx])
        total_error += error
        data_count += 1
    if data_count == 0:
        return np.nan  # Return NaN if count is zero to avoid division by zero
    return total_error / data_count

def calculate_mean_abs_percentage_error(demand_data, forecast_data, start_yr, end_yr, question=1):
    total_error = 0
    data_count = 0
    for t_idx in range(len(demand_data)):
        if forecast_data[t_idx] is None or np.isnan(forecast_data[t_idx]):
            continue
        if check_valid_time(start_yr, end_yr, t_idx, question):
            continue
        error = demand_data[t_idx] - forecast_data[t_idx]
        if demand_data[t_idx] != 0:  # Check if demand value is not zero to avoid division by zero
            total_error += abs(error / demand_data[t_idx])
            data_count += 1
    if data_count == 0:
        return np.nan  # Return NaN if count is zero to avoid division by zero
    return 100 * total_error / data_count

def calculate_root_mean_squared_error(demand_data, forecast_data, start_yr, end_yr, question=1):
    residuals = compute_residuals(demand_data, forecast_data, start_yr, end_yr, question)
    total_squared_error = 0
    data_count = 0
    for error in residuals:
        data_count += 1
        total_squared_error += error ** 2
    if data_count == 0:
        return np.nan  # or any other value to indicate that RMSE cannot be calculated
    return np.sqrt(total_squared_error / data_count)

def simple_exponential_smoothing(series, alpha_val):
    smoothed_series = pd.Series(index=series.index)
    smoothed_series[0] = series[0]
    for t_idx in range(1, len(series)):
        smoothed_series[t_idx] = alpha_val * series[t_idx] + (1 - alpha_val) * smoothed_series[t_idx - 1]
    return smoothed_series

def check_valid_time(start_yr, end_yr, t_idx, question):
    """
    Check if the time index is within the valid range.
    """
    # Placeholder logic, modify as needed
    return False

def compute_residuals(demand_data, forecast_data, start_yr, end_yr, question=1):
    """
    Compute residuals for error calculation.
    """
    # Placeholder logic, modify as needed
    return []

# Initialize lists to store error metrics
rmse_errors = []
mape_errors = []
mae_errors = []

# Iterate over alpha values from 0.1 to 1.0
for i, alpha in enumerate(range(1, 11)):
    alpha /= 10  # Convert alpha to float
    smoothed_series = simple_exponential_smoothing(U_style, alpha)

    # Calculate error metrics for the current alpha
    rmse = calculate_root_mean_squared_error(U_style, smoothed_series, 2019, 2022)
    mape = calculate_mean_abs_percentage_error(U_style, smoothed_series, 2019, 2022)
    mae = calculate_mean_abs_error(U_style, smoothed_series, 2019, 2022)

    # Append error metrics to respective lists
    rmse_errors.append(rmse)
    mape_errors.append(mape)
    mae_errors.append(mae)

# Find the alpha values corresponding to minimum error metrics
min_rmse_alpha = (rmse_errors.index(min(rmse_errors)) + 1) / 10
min_mape_alpha = (mape_errors.index(min(mape_errors)) + 1) / 10
min_mae_alpha = (mae_errors.index(min(mae_errors)) + 1) / 10

# Find the minimum error metrics
min_rmse = min(rmse_errors)
min_mape = min(mape_errors)
min_mae = min(mae_errors)

# Print the results
print("Minimum RMSE:", min_rmse, "- Alpha:", min_rmse_alpha)
print("Minimum MAPE:", min_mape, "- Alpha:", min_mape_alpha)
print("Minimum MAE:", min_mae, "- Alpha:", min_mae_alpha)

# Set alpha_star to the alpha corresponding to minimum MAE
alpha_star = min_mae_alpha


# # QUESTION 2

# # PART A

temp_df = pd.read_csv("domestic_beer_sales.csv", header=None)

df = pd.DataFrame()
df["Demand"] = temp_df.iloc[:, 1]  

dates = pd.date_range(start="2010-01-01", end="2014-12-31", freq="M")
df['Date'] = dates
df['Date'] = df['Date'].dt.year  
demand = df["Demand"]
print(df['Date'])

plt.figure(figsize=(10, 5))
plt.plot(demand)  
plt.xticks(range(0, len(df['Date']), 12), df['Date'].unique(), rotation=45)
plt.title('Domestic Beer Sales by Month Between 2010 and 2014')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.tight_layout()
plt.show()


# # PART B & C

# Tau = 1

naive_forecast = df["Demand"].shift(1)
error = df["Demand"] - naive_forecast

mae = np.mean(np.abs(error))
mape = np.mean(np.abs(error / df["Demand"])) * 100
rmse = np.sqrt(np.mean(error**2))

print("Mean Absolute Error (MAE):", mae)
print("Mean Absolute Percentage Error (MAPE):", mape)
print("Root Mean Squared Error (RMSE):", rmse)

plt.figure(figsize=(14, 5))
plt.plot(df['Demand'], label='Actual Sales')
plt.plot(naive_forecast, label='Naive Forecast')
plt.xticks(range(0, len(df['Date']), 12), df['Date'].unique(), rotation=45)
plt.title('Domestic Beer Sales by Month Between 2010 and 2014 with Naive Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.tight_layout()
plt.show()

bijan.eplot(df["Demand"][1:]-naive_forecast[1:])

naive_forecast = df["Demand"].shift(12)
error = df["Demand"] - naive_forecast
mae = np.mean(np.abs(error))
mape = np.mean(np.abs(error / df["Demand"])) * 100
rmse = np.sqrt(np.mean(error**2))

print("Mean Absolute Error (MAE):", mae)
print("Mean Absolute Percentage Error (MAPE):", mape)
print("Root Mean Squared Error (RMSE):", rmse)

plt.figure(figsize=(14, 5))
plt.plot(df['Demand'], label='Actual Sales')
plt.plot(naive_forecast, label='Naive Forecast')
plt.xticks(range(0, len(df['Date']), 12), df['Date'].unique(), rotation=45)
plt.title('Domestic Beer Sales by Month Between 2010 and 2014 with Naive Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.tight_layout()
plt.show()

bijan.eplot(df["Demand"][12:]-naive_forecast[12:])


# # PART C

# For tau = 1


data = pd.read_csv("domestic_beer_sales.csv", usecols=[0, 1], header=None)
data.columns = ['Month', 'Sales']

data['Forecast'] = data["Sales"].shift(1)

data.dropna(inplace=True)



def res_analysis(months, sales, forecast, plots=True):
    residuals = forecast - sales
    std_res = (residuals - residuals.mean()) / residuals.std()
    
    mean_residuals = residuals.mean()
    mean_sales = sales.mean()
    mean_residuals_percentage = (mean_residuals / mean_sales) * 100
    
    if plots:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))
        axs[0][0].plot(months, residuals)
        axs[0][0].set_xlabel('Month')
        axs[0][0].set_ylabel('Standardized Residuals')
        axs[0][0].grid(True)
        axs[0][0].hlines(0, min(months), max(months), 'r')

        kde = sm.nonparametric.KDEUnivariate(residuals)
        kde.fit()  # Estimate the densities
        axs[1][0].hist(residuals)
        ax_kde = axs[1][0].twinx()
        ax_kde.plot(kde.support, kde.density, 'r')

        sm.qqplot(std_res, ax=axs[0][1])
        sm.qqline(axs[0][1], "45")
        sm.graphics.tsa.plot_acf(std_res, ax=axs[1][1])

    print(f'Mean of Residual: {mean_residuals:.4f}')
    print(f'S.D. of Residual: {residuals.std():.4f}')
    print(f'MAE : {abs(residuals).mean():.4f}')
    print(f'MAPE: {(abs(residuals) / sales * 100).mean():.4f}')
    print(f'MSE : {sm.tools.eval_measures.mse(sales, forecast):.4f}')
    print(f'RMSE: {sm.tools.eval_measures.rmse(sales, forecast):.4f}')
    print(f'Mean of Residual / Mean of Sales: {mean_residuals_percentage:.2f}%')
    
    return residuals, std_res


months_a = data['Month'][3:].reset_index(drop=True)
sales_a = data['Sales'][3:].reset_index(drop=True)
forecast_a = data['Forecast'][:-3].reset_index(drop=True)


_ = res_analysis(months_a, sales_a, forecast_a, plots=True)


# For Tau = 12


data = pd.read_csv("domestic_beer_sales.csv", usecols=[0, 1], header=None)
data.columns = ['Month', 'Sales']

data['Forecast'] = data["Sales"].shift(12)

data.dropna(inplace=True)


def res_analysis(months, sales, forecast, plots=True):
    residuals = forecast - sales
    std_res = (residuals - residuals.mean()) / residuals.std()
    
    mean_residuals = residuals.mean()
    mean_sales = sales.mean()
    mean_residuals_percentage = (mean_residuals / mean_sales) * 100
    
    if plots:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))
        axs[0][0].plot(months, residuals)
        axs[0][0].set_xlabel('Month')
        axs[0][0].set_ylabel('Standardized Residuals')
        axs[0][0].grid(True)
        axs[0][0].hlines(0, min(months), max(months), 'r')

        kde = sm.nonparametric.KDEUnivariate(residuals)
        kde.fit()  # Estimate the densities
        axs[1][0].hist(residuals)
        ax_kde = axs[1][0].twinx()
        ax_kde.plot(kde.support, kde.density, 'r')

        sm.qqplot(std_res, ax=axs[0][1])
        sm.qqline(axs[0][1], "45")
        sm.graphics.tsa.plot_acf(std_res, ax=axs[1][1])

    print(f'Mean of Residual: {mean_residuals:.4f}')
    print(f'S.D. of Residual: {residuals.std():.4f}')
    print(f'MAE : {abs(residuals).mean():.4f}')
    print(f'MAPE: {(abs(residuals) / sales * 100).mean():.4f}')
    print(f'MSE : {sm.tools.eval_measures.mse(sales, forecast):.4f}')
    print(f'RMSE: {sm.tools.eval_measures.rmse(sales, forecast):.4f}')
    print(f'Mean of Residual / Mean of Sales: {mean_residuals_percentage:.2f}%')
    
    return residuals, std_res


months_a = data['Month'][3:].reset_index(drop=True)
sales_a = data['Sales'][3:].reset_index(drop=True)
forecast_a = data['Forecast'][:-3].reset_index(drop=True)


_ = res_analysis(months_a, sales_a, forecast_a, plots=True)


# # PART D


def find_c_t(gamma: float,
             N: int,
             results: pd.DataFrame,
             t: int) -> float:
    """
    Calculate c_t
    :param gamma: float, smoothing parameter for seasonality
    :param N: int, number of seasonal periods
    :param results: pd.DataFrame, results of the Holt-Winters model
    :param t: int, index of the current period
    :return: float, c_t
    """
    # initialization rule
    if t <= N:
        return results.loc[t, 'D_t'] / np.mean(results.loc[:N, 'D_t'])
    # normal calculation
    else:
        return gamma * (results.loc[t, 'D_t'] / results.loc[t, 'S_t']) + (1 - gamma) * results.loc[t - N, 'c_t']


def find_S_t(alpha: float,
             N: int,
             results: pd.DataFrame,
             t: int) -> float:
    """
    Calculate S_t
    :param alpha: float, smoothing parameter for level
    :param N: int, number of seasonal periods
    :param results: pd.DataFrame, results of the Holt-Winters model
    :param t: int, index of the current period
    :return: float, S_t
    """
    # initialization rule
    if t == N + 1:
        return results.loc[t, 'D_t'] / results.loc[t - N, 'c_t']
    # normal calculation
    else:
        return alpha * (results.loc[t, 'D_t'] / results.loc[t - N, 'c_t']) + (1 - alpha) * (results.loc[t - 1, 'S_t'] + results.loc[t - 1, 'G_t'])


def find_G_t(beta: float,
             N: int,
             results: pd.DataFrame,
             t: int) -> float:
    """
    Calculate G_t
    :param beta: float, smoothing parameter for trend
    :param N: int, number of seasonal periods
    :param results: pd.DataFrame, results of the Holt-Winters model
    :param t: int, index of the current period
    :return: float, G_t
    """
    # initialization rule
    if t == N + 1:
        return results.loc[t, 'S_t'] - (results.loc[t - 1, 'D_t'] / results.loc[t - 1, 'c_t'])
    # normal calculation
    else:
        return beta * (results.loc[t, 'S_t'] - results.loc[t - 1, 'S_t']) + (1 - beta) * results.loc[t - 1, 'G_t']


def find_F_t(tau: int,
             N: int,
             results: pd.DataFrame,
             t: int,
             data_len: int) -> float:
    """
    Calculate F_t
    :param results: pd.DataFrame, results of the Holt-Winters model
    :param t: int, index of the current period
    :param N: int, number of seasonal periods
    :param tau: int, forecast horizon
    :param data_len: int, length of the time series
    :return: float, F_t
    """

    c_index = t - N
    if c_index > data_len:
        c_index = data_len

    S_index = t - tau
    if S_index > data_len:
        S_index = data_len

    G_index = t - tau
    if G_index > data_len:
        G_index = data_len

    return ((results.loc[S_index, 'S_t'] + tau * results.loc[G_index, 'G_t']) *
            results.loc[c_index, 'c_t'])


def read_data(file_path):
    return pd.read_csv(file_path, header=None, names=["Date", "Demand"])

def preprocess_data(data):
    data['Year'] = pd.to_datetime(data['Date']).dt.year
    return data['Demand']

def mean_absolute_error(demand, forecast, start_year, end_year, question=1):
    tot = 0
    count = 0
    for t in range(len(demand)):
        error = get_abs_error(demand, forecast, t, start_year, end_year, question)
        if error is None:
            continue
        count += 1
        tot += error
    return tot / count

def calculate_mape(demand, forecast, start_year, end_year, q=1):
    total_error = 0
    count = 0
    for period in range(len(demand)):
        error = get_error(demand, forecast, period, start_year, end_year, q)
        if error is None:
            continue
        count += 1
        total_error += abs(error / demand.loc[period])
    return 100 * total_error / count

def calculate_mae(demand, forecast, start_year, end_year, q=1):
    total_error = 0
    count = 0
    for period in range(len(demand)):
        error = get_error(demand, forecast, period, start_year, end_year, q)
        if error is None:
            continue
        count += 1
        total_error += error
    return total_error / count

def calculate_rmse(demand, forecast, start_year, end_year):
    total_squared_error = 0
    count = 0
    start_period = get_start(start_year)
    end_period = min(get_end(end_year), len(forecast) - 1)
    for period in range(start_period, end_period + 1):
        error = get_error(demand, forecast, period, start_year, end_year)
        if error is not None:
            total_squared_error += error ** 2
            count += 1
    mean_squared_error = total_squared_error / count
    return np.sqrt(mean_squared_error)

def get_error(demand, forecast, period, start_year, end_year, q=1):
    start_period = get_start(start_year, q)
    end_period = min(get_end(end_year, q), len(forecast) - 1)
    
    if period < start_period or period > end_period:
        return None
    
    demand_index = period - start_period  # Adjust index for demand data
    if demand_index < 0 or demand_index >= len(demand):
        return None
    
    return abs(demand.iloc[demand_index] - forecast[demand_index])



def triple_exp_forecast(demand, alpha, beta, gamma, N, start):
    results = []
    for t in range(len(demand)):
        if t == 0:
            s_t = demand[0]
            c_t = demand[1] - demand[0]
            g_t = c_t
            f_t = demand[0]  # Initialize f_t with the first demand value
        elif t < N:
            s_t = alpha * (demand[t] / c_t) + (1 - alpha) * (results[t-1][0] + results[t-1][2])
            g_t = beta * (s_t - results[t-1][0]) + (1 - beta) * results[t-1][2]
            f_t = demand[t]  # Initialize f_t with the current demand value
        else:
            if t - N < 0:
                f_t = demand[t]  # or any other appropriate value
            else:
                f_t = (results[t-N][0] + results[t-N][2]) * c_t
            s_t = alpha * (demand[t] / c_t) + (1 - alpha) * (results[t-N][0] + results[t-N][2])
            g_t = beta * (s_t - results[t-N][0]) + (1 - beta) * results[t-N][2]
        c_t = gamma * (demand[t] / s_t) + (1 - gamma) * c_t
        results.append((s_t, c_t, g_t, f_t))
    return [r[3] for r in results]


def initialize(demand, N):
    return demand[:N].mean(), demand[N] - demand[N-1], demand[N]

def get_end(year, q=1):
    if q == 1:
        return (year - 2013) * 12 + 11
    return (year - 2010) * 12 + 11

def get_start(year, q=1):
    if q == 1:
        return (year - 2013) * 12
    return (year - 2010) * 12


if __name__ == "__main__":
    file_path = "domestic_beer_sales.csv"
    data_beer = read_data(file_path)
    demand_beer = preprocess_data(data_beer)

    N = 12
    best_mape_beer = float('inf')
    best_mae_beer = float('inf')
    best_rmse_beer = float('inf')
    best_alpha_mape_beer = None
    best_beta_mape_beer = None
    best_gamma_mape_beer = None
    best_alpha_mae_beer = None
    best_beta_mae_beer = None
    best_gamma_mae_beer = None
    best_alpha_rmse_beer = None
    best_beta_rmse_beer = None
    best_gamma_rmse_beer = None
    start_period_beer = 2019

    for alpha_beer in np.arange(0.1, 1.1, 0.1):
        for beta_beer in np.arange(0.1, 1.1, 0.1):
            for gamma_beer in [0.1, 0.5, 0.9]:
                forecast_beer = triple_exp_smoothing(alpha_beer, beta_beer, gamma_beer, N, 1, demand_beer.to_list())

                mape_beer = mean_abs_percentage_error(demand_beer, forecast_beer, 2012, 2014, 2)
                rmse_beer = root_mean_squared_error(demand_beer, forecast_beer, 2012, 2014, 2)
                mae_beer = mean_absolute_error(demand_beer, forecast_beer, 2012, 2014, 2)

                if mape_beer < best_mape_beer:
                    best_mape_beer, best_mape_forecast_beer, best_alpha_mape_beer, best_beta_mape_beer, best_gamma_mape_beer = mape_beer, forecast_beer, alpha_beer, beta_beer, gamma_beer
                if mae_beer < best_mae_beer:
                    best_mae_beer, best_mae_forecast_beer, best_alpha_mae_beer, best_beta_mae_beer, best_gamma_mae_beer = mae_beer, forecast_beer, alpha_beer, beta_beer, gamma_beer
                if rmse_beer < best_rmse_beer:
                    best_rmse_beer, best_rmse_forecast_beer, best_alpha_rmse_beer, best_beta_rmse_beer, best_gamma_rmse_beer = rmse_beer, forecast_beer, alpha_beer, beta_beer, gamma_beer

    print("Best MAPE (Beer):", best_mape_beer)
    print("Alpha (MAPE):", best_alpha_mape_beer)
    print("Beta (MAPE):", best_beta_mape_beer)
    print("Gamma (MAPE):", best_gamma_mape_beer)

    print("Best RMSE (Beer):", best_rmse_beer)
    print("Alpha (RMSE):", best_alpha_rmse_beer)
    print("Beta (RMSE):", best_beta_rmse_beer)
    print("Gamma (RMSE):", best_gamma_rmse_beer)

    print("Best MAE (Beer):", best_mae_beer)
    print("Alpha (MAE):", best_alpha_mae_beer)
    print("Beta (MAE):", best_beta_mae_beer)
    print("Gamma (MAE):", best_gamma_mae_beer)


plt.figure(figsize=(14, 5), facecolor='lightgrey')
plt.plot(demand_beer, label="Demand (Beer)")

plt.plot(best_mape_forecast_beer, label=f"Best MAPE Forecast", linestyle='-')
plt.plot(best_mae_forecast_beer, label=f"Best MAE Forecast", linestyle='-')
plt.plot(best_rmse_forecast_beer, label=f"Best RMSE Forecast", linestyle='-')

plt.title("Beer Demand and Best MAPE, MAE, RMSE)")
plt.xlabel("Monthly Basis")
plt.ylabel("Demand Basis")
plt.legend()
plt.grid(True)
plt.show()

# Function to generate a list of prediction intervals for beer forecast
def get_interval_list_beer(forecast_beer, confidence_level, rmse_beer, start_year=2012, end_year=2014):
    forecast_beer = forecast_beer[get_start(start_year): get_end(end_year) + 1]
    return [get_interval(f, confidence_level, rmse_beer) for f in forecast_beer]


optimal_forecast_mape_beer = triple_exp_forecast(demand_beer, best_alpha_mape_beer, best_beta_mape_beer, best_gamma_mape_beer, N, start_period_beer)
optimal_forecast_mae_beer = triple_exp_forecast(demand_beer, best_alpha_mae_beer, best_beta_mae_beer, best_gamma_mae_beer, N, start_period_beer)
optimal_forecast_rmse_beer = triple_exp_forecast(demand_beer, best_alpha_rmse_beer, best_beta_rmse_beer, best_gamma_rmse_beer, N, start_period_beer)

alpha_star_beer = best_alpha_rmse_beer
beta_star_beer = best_beta_rmse_beer
gamma_star_beer = best_gamma_rmse_beer
std_estimator_beer = calculate_rmse(demand_beer, optimal_forecast_rmse_beer, 2011, 2013)
prediction_intervals_beer = get_interval_list(optimal_forecast_rmse_beer, 0.9, std_estimator_beer, 2014, 2)

lower_bound_beer = [interval[0] for interval in prediction_intervals_beer]
upper_bound_beer = [interval[1] for interval in prediction_intervals_beer]

if prediction_intervals_beer and optimal_forecast_rmse_beer:
    print(f"\n90% Confidence Prediction Intervals for Beer:")
    for interval, forecast in zip(prediction_intervals_beer, optimal_forecast_rmse_beer):
        print(f"Forecast: {forecast}, Interval: {interval}")
else:
    print("No data to display.")


plt.figure(figsize=(10, 5))
plt.plot(demand_beer, color="black", label='Demand')

# Assuming demand_beer is indexed by time
plt.fill_between(range(4 * 12, len(lower_bound_beer) + 4 * 12 ), lower_bound_beer, upper_bound_beer, color='red', alpha=0.3, label='Prediction Intervals')

# Adjust x-axis ticks to represent years
plt.xticks(range(0, len(demand_beer), 12), range(2014, 2014 + len(demand_beer) // 12), rotation=45)
plt.title('Beer Demand Over Time with 90% Prediction Intervals')
plt.xlabel('Year')
plt.ylabel('Demand')
plt.legend()
plt.show()


# # PART E

def find_coeff_alpha(alpha_param: float, num_seasons: int, results_df: pd.DataFrame, t_index: int) -> float:
    """
    Calculate the coefficient alpha (level smoothing parameter).
    """
    if t_index == num_seasons + 1:
        return results_df.loc[t_index, "D_t"] / results_df.loc[t_index - num_seasons, "c_t"]
    else:
        return alpha_param * (results_df.loc[t_index, "D_t"] / results_df.loc[t_index - num_seasons, "c_t"]) + (
            1 - alpha_param
        ) * (results_df.loc[t_index - 1, "S_t"] + results_df.loc[t_index - 1, "G_t"])

def find_coeff_beta(beta_param: float, num_seasons: int, results_df: pd.DataFrame, t_index: int) -> float:
    """
    Calculate the coefficient beta (trend smoothing parameter).
    """
    if t_index == num_seasons + 1:
        return results_df.loc[t_index, "S_t"] - (
            results_df.loc[t_index - 1, "D_t"] / results_df.loc[t_index - 1, "c_t"]
        )
    else:
        return (
            beta_param * (results_df.loc[t_index, "S_t"] - results_df.loc[t_index - 1, "S_t"])
            + (1 - beta_param) * results_df.loc[t_index - 1, "G_t"]
        )

def calculate_forecast(tau: int, num_seasons: int, results_df: pd.DataFrame, t_index: int, data_length: int) -> float:
    """
    Calculate the forecast F_t.
    """
    c_index = t_index - num_seasons
    if c_index > data_length:
        c_index = data_length

    S_index = t_index - tau
    if S_index > data_length:
        S_index = data_length

    G_index = t_index - tau
    if G_index > data_length:
        G_index = data_length

    return (
        results_df.loc[S_index, "S_t"] + tau * results_df.loc[G_index, "G_t"]
    ) * results_df.loc[c_index, "c_t"]

def triple_exp_smoothing(alpha, beta, gamma, N, max_forecast_horizon, data):
    """
    Holt-Winters Triple Exponential Smoothing (Multiplicative)
    :param alpha: float, smoothing parameter for level
    :param beta: float, smoothing parameter for trend
    :param gamma: float, smoothing parameter for seasonality
    :param N: int, number of seasonal periods
    :param max_forecast_horizon: int, maximum number of periods to forecast
    :param data: list, time series to forecast
    :return: pd.Series, forecast
    """

    # create a dataframe to store the results
    # dataframe has 6 columns: t, D_t, S_t, G_t, c_t, F_t
    results = pd.DataFrame(
        columns=["t", "D_t", "S_t", "G_t", "c_t", "F_t"], index=range(1, len(data) + 1)
    )

    # put the time series into a dataframe
    results["t"] = range(1, len(data) + 1)
    results["D_t"] = data

    length_of_data = len(data)

    for t in results["t"].tolist():

        # if 't' is less than N, we should first initialize c_t
        if t <= N:
            results.loc[t, "c_t"] = find_c_t(gamma, N, results, t)

        # if 't' is equal to N, we should first initialize S_t and G_t
        elif t == N + 1:
            results.loc[t, "S_t"] = find_S_t(alpha, N, results, t)
            results.loc[t, "G_t"] = find_G_t(beta, N, results, t)
            results.loc[t, "c_t"] = find_c_t(gamma, N, results, t)

        # if 't' is greater than N, we can calculate S_t, G_t, and c_t
        # and then calculate F_t
        else:
            results.loc[t, "S_t"] = find_S_t(alpha, N, results, t)
            results.loc[t, "G_t"] = find_G_t(beta, N, results, t)
            results.loc[t, "c_t"] = find_c_t(gamma, N, results, t)
            results.loc[t, "F_t"] = find_F_t(1, N, results, t, length_of_data)

    # add forecast values to the results dataframe
    for tau in range(1, max_forecast_horizon + 1):
        new_row = [len(data) + tau + 1, np.nan, np.nan, np.nan, np.nan, np.nan]
        results.loc[len(data) + tau + 1] = new_row

        results.loc[len(data) + tau + 1, "F_t"] = find_F_t(
            tau, N, results, len(data) + tau, length_of_data
        )
    results["t"] = results["t"].astype(int)

    forecasts = results["F_t"].tolist()

    return forecasts

def calculate_rmse(demand, forecast, start_year, end_year):
    total_squared_error = 0
    count = 0
    for period in range(get_start(start_year), get_end(end_year) + 1):
        error = get_error(demand, forecast, period, start_year, end_year)
        if error is not None:
            total_squared_error += error ** 2
            count += 1
    if count == 0:
        return np.nan  # Return NaN if count is zero to avoid division by zero
    return np.sqrt(total_squared_error / count)

def calculate_mape(demand, forecast, start_year, end_year):
    total_abs_percentage_error = 0
    count = 0
    for period in range(len(demand)):
        error = get_error(demand, forecast, period, start_year, end_year)
        if error is None:
            continue
        total_abs_percentage_error += abs(error / demand[period])
        count += 1
    if count == 0:
        return np.nan  # Return NaN if count is zero to avoid division by zero
    return 100 * (total_abs_percentage_error / count)

def calculate_mae(demand, forecast, start_year, end_year):
    total_abs_error = 0
    count = 0
    for period in range(len(demand)):
        error = get_error(demand, forecast, period, start_year, end_year)
        if error is None:
            continue
        total_abs_error += abs(error)
        count += 1
    if count == 0:
        return np.nan  # Return NaN if count is zero to avoid division by zero
    return total_abs_error / count

def get_error(demand, forecast, period, start_year, end_year):
    if period < start_year or period > end_year:
        return None
    return abs(demand[period] - forecast[period])


for tau_value in [3, 6]:
    forecast_result = holt_winters_triple_exp_smoothing(best_alpha_mape_beer, best_beta_mape_beer, best_gamma_mape_beer, N, tau_value, demand_beer.to_list())
    mape = calculate_mape(demand_beer, forecast_result, 2012, 2014)
    mae = calculate_mae(demand_beer, forecast_result, 2012, 2014)
    rmse = calculate_rmse(demand_beer, forecast_result, 2012, 2014)
    print(f"Performance Metrics for {tau_value} Month Lookahead Forecast")
    print("Mean Absolute Percentage Error (MAPE):", mape)
    print("Mean Absolute Error (MAE):", mae)
    print("Root Mean Square Error (RMSE):", rmse)
    
    plt.figure(figsize=(10, 5), facecolor='lightgrey')
    plt.plot(demand_beer, color="black", label="Demand")
    plt.plot(forecast_result, color="red", label=f"Forecast of tau = {tau_value}")
    plt.title("Demand and Forecast")
    plt.xlabel("Month")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(True)
    plt.show()
    


# @credit Sarp Çağan Kelleci - @Tan Karahasanoğlu
