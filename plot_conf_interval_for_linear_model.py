# Resources 
# https://stackoverflow.com/questions/61292464/get-confidence-interval-from-sklearn-linear-regression-in-python
# https://github.com/statsmodels/statsmodels/issues/987
# https://stackoverflow.com/questions/17559408/confidence-and-prediction-intervals-with-statsmodels

alpha = 1 - 95/100 
print(f'For 95% conf interval {alpha=}') # 0.05

# pip install statsmodels
import statsmodels.api as sm

# exmaple from: https://online.stat.psu.edu/stat415/lesson/7/7.5
dataset = np.array([[190,	7.23],
                    [160,	8.53],
                    [134,	9.82],
                    [129,	10.26],
                    [172,	8.96],
                    [197,	12.27],
                    [167,	10.28],
                    [239,	4.45],
                    [542,	1.87],
                    [372,	4.00],
                    [245,	3.30],
                    [376,	4.30],
                    [454,	0.80],
                    [410,	0.50]]
)

x_train_plot = dataset[:, 1].reshape(-1,1)
x_train = sm.add_constant(x_train_plot)
y_train = dataset[:, 0]

# Linear Regression Model 
lr = sm.OLS(y_train, x_train).fit()
conf_interval = lr.conf_int(alpha)

conf_interval_df = pd.DataFrame(conf_interval, columns=['lower', 'upper'])
print(conf_interval_df)
print('params=', lr.params)
xtest_plot = np.arange(0, 13, 0.1).reshape(-1,1)
xtest = sm.add_constant(xtest_plot)
predictions = lr.get_prediction(xtest)
predictions_frame = predictions.summary_frame(alpha=alpha)

# Plot the data 
plt.scatter(x_train_plot, y_train, label='data')
plt.plot(xtest_plot, predictions_frame['mean'], label='mean pred')
plt.plot(xtest_plot, predictions_frame['mean_ci_lower'], 'r--', label='mean_ci_lower')
plt.plot(xtest_plot, predictions_frame['mean_ci_upper'], 'b--', label='mean_ci_upper')

plt.plot(xtest_plot, predictions_frame['obs_ci_lower'], 'g--', label='obs_ci_lower')
plt.plot(xtest_plot, predictions_frame['obs_ci_upper'], 'k--', label='obs_ci_upper')
plt.title('Confidence Interval')
plt.xlabel('Catch')
plt.ylabel('Price')
plt.legend()
plt.show()
