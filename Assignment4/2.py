import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression



# creating DataFrame
df = pd.read_csv('Weight-Height.csv', nrows=30)

# prepare the data
x = df[['Weight']]
y = df[['Height']]


# mean of x and y
xmean = np.mean(x)
ymean = np.mean(y)

# create and train the model
model = LinearRegression()
model.fit(x,y)

# predict the weight
yhat = model.predict(x)

# plot the data
plt.scatter(x,y)
plt.scatter(xmean, ymean, color='red')
plt.plot(x, yhat, color='blue')
plt.title('Scatter plot of weight and length')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()

# Now Model Evaluation using RMSE and R2 Score
rmse = print(f'RMSE: ', np.sqrt(metrics.mean_squared_error(y, yhat)))
r2 = print(f'R2 score: ', metrics.r2_score(y, yhat))