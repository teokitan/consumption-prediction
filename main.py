import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import metrics
import matplotlib.pyplot as mat

df = pd.read_csv("auto-mpg.data.csv")

# Displacement has 0.933 correlation
#del df['displacement']

# MPG in American, DELETE, and use "mpgl" - liters per 100 km
del df['mpg']

# Split train and test
feature_col_names = ['mpgl', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']
predicted_class_names = ['mpgl']

X = df[feature_col_names].values
Y = df[predicted_class_names].values

split_test_size = 0.30

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=split_test_size, random_state=42)

# Replace missing values
fill_0 = SimpleImputer(missing_values=-69, strategy="mean")
X_train = fill_0.fit_transform(X_train)
X_test = fill_0.fit_transform(X_test)

# Implement algorithm
from sklearn.linear_model import Ridge
nb_model = Ridge(max_iter=10000, random_state=42)
nb_model.fit(X_train, Y_train)

nb_predict_train = nb_model.predict(X_train)
nb_predict_test = nb_model.predict(X_test)

# Report loss
res = metrics.explained_variance_score(Y_test, nb_predict_test)
print(res)

# Plot the results
mat.plot(Y_test, color="skyblue", label="actual")
mat.plot(nb_predict_test, color="red", label="predicted")
mat.xlabel("data")
mat.ylabel("consumption")
mat.title("Predicted vs actual data")
mat.legend()
mat.show()

# End the program
print("END")