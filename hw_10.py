import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Загрузка данных
data_path = 'realty_data.csv'
data = pd.read_csv(data_path)

# Предобработка данных
data = data[["total_square", "rooms", "price"]].dropna()
data = data.apply(pd.to_numeric, errors='coerce').dropna()

# Разделение данных на признаки и целевую переменную
X = data[["total_square", "rooms"]]
y = data["price"]

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

# Оценка модели
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Среднеквадратичная ошибка: {mse}")

# Сохранение модели
joblib.dump(model, 'realty_model.pkl')
