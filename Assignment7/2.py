import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

# read dataset into DataFrame;
df = pd.read_csv("suv.csv")
print(df)


# define feature variables as "Age" and "EStimatedSalary";
X = df[["Age", "EstimatedSalary"]]


# define target variable as "Purchased";
y = df["Purchased"]


# split dataset into 80 and 20;
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# scale the features using standard scaler;
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# train the model by using DecisionTreeClassifier with "entropy";
dt_entropy = DecisionTreeClassifier(criterion="entropy")
dt_entropy.fit(X_train, y_train)
y_pred_entropy = dt_entropy.predict(X_test)

# compute confusion matrix and classification report;
dt_entropy_conf = confusion_matrix(y_test, y_pred_entropy)
print(f"confusion_matrix: {dt_entropy_conf}")

dt_entropy_class = classification_report(y_test, y_pred_entropy)
print(f"classification_report:{dt_entropy_class}")

# train the model from "entropy" to "gini";
dt_gini = DecisionTreeClassifier(criterion="gini")
dt_gini.fit(X_train, y_train)
y_pred_gini = dt_gini.predict(X_test)

# compute confusion matrix and classification report;
dt_gini_conf = confusion_matrix(y_test, y_pred_gini)
print(f"confusion_matrix gini: {dt_gini_conf}")

dt_gini_class = classification_report(y_test, y_pred_gini)
print(f"classification_report gini: {dt_gini_class}")




""""
Here, entropy produced 6 false positives, and 6 false negatives. Whereas, gini produced 6 false positives and 4 false negatives. From this it is seen that gini is better at identifying class 1.
In case of class 0, both the models have performed identically producing 6 false positives.

Overall gini has performed better as shown by 88% accuracy compared to that of Entropy's 85%.

Gini is more precise for negative (0) class and for class 1, it is slightly better than entropy (0.79 vs 0.78)

"""

