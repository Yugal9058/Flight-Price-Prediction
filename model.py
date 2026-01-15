import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

df = pd.read_csv("flight_data.csv")

df["Journey_day"] = pd.to_datetime(df["Date_of_Journey"]).dt.day
df["Journey_month"] = pd.to_datetime(df["Date_of_Journey"]).dt.month
df.drop("Date_of_Journey", axis=1, inplace=True)

df["Duration"] = df["Duration"].str.replace("h", "*60").str.replace("m", "+").apply(eval)

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("R2 Score:", r2_score(y_test, model.predict(X_test)))

pickle.dump(model, open("flight_fare_model.pkl", "wb"))
