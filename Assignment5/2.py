import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score

# Loading the data from csv file
data = pd.read_csv('50_Startups.csv')
# print(data)
'''
The variables inside the dataset are: R&D Spend', 'Administration', 'Marketing Spend', 'State', and 'Profit'
'''

# droping the state column to create dataframe
df = data.drop(columns='State')
# print(df)

# calculating correlations and display correlation using heatmap
correlation = df.corr().round(2)
print(correlation)
sns.heatmap(data=correlation, annot=True)
plt.show()

'''
From the heatmap, R&D Spend shows a strong correlation with profit (0.97) and Marketing Spend shows a moderate correlation with profit (0.75). However, Administration shows almost no correlation (0.20). Therefore, for our model, I will select two features (R&D Spend, and Marketing Spend) as independent variables and Profit as the target variable.
'''

# selecting 'R&D Spend' and 'Marketing Spend' independent variables and 'profit' as the dependent variable
x = df[['R&D Spend', "Marketing Spend"]]
y = df[['Profit']]



# Ploting explanatory variables against profit
plt.subplot(1, 2, 1)
plt.scatter(df['R&D Spend'], df['Profit'])
plt.xlabel('R&D Spend')
plt.ylabel('Profit')

plt.subplot(1, 2, 2)
plt.scatter(df['Marketing Spend'], df['Profit'])
plt.xlabel('Marketing Spend')
plt.ylabel('Profit')
plt.tight_layout()
plt.show()

# spliting the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)
# print("Training data size:", x_train.shape)
# print("Testing data size:", x_test.shape)

# initializing and train the linear regression model with training data
model = LinearRegression()
model.fit(x_train, y_train)

# evaluating the model on training data
y_train_predict = model.predict(x_train)
rmse_train = root_mean_squared_error(y_train, y_train_predict)
r2_train = r2_score(y_train, y_train_predict)
print(f"RMSE and R2 (training data): {rmse_train}, {r2_train}")

# evaluating the model on testing data
y_test_predict = model.predict(x_test)
rmse_test = root_mean_squared_error(y_test, y_test_predict)
r2_test = r2_score(y_test, y_test_predict)
print(f"RMSE and R2 (testing data): {rmse_test}, {r2_test}")

'''
The model's performance looks pretty good with RMSE of 9358.58 and 94.36% respectively on training data and 7073.85 and 96.84% on testing data. Lower value of RMSE on testing data indicates that the model is not overfitting.
'''