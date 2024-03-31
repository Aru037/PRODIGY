import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")
print(df)

features = ["GrLivArea", "BedroomAbvGr", "FullBath"]
X = df[features]
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 42)

print("X_train has {} rows".format(X_train.shape[0]))
print("X_test has {} rows".format(X_test.shape[0]))

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices")
plt.show()

new_data = pd.DataFrame({'GrLivArea' : [2000], 'BedroomAbvGr' : [3], 'FullBath' : [2]})
predicted_price = model.predict(new_data)
print(f"Predicted Price: {predicted_price[0]}")