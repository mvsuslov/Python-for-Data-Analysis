import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor


# получаем данные из файла
data = pd.read_csv('realty_data.csv')

# проверка на вылет значений
data = data[["total_square", "rooms", "price"]].dropna()
data = data[(data['total_square'] > 0) & (data['price'] > 0)]

# берем два признака: площадь и количество комнат
X = data[["total_square", "rooms"]]  
y = data["price"]

# разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Оценка модели
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# print(f"Среднеквадратичная ошибка: {mse}")

# scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
# print(f"Среднеквадратичная ошибка по кросс-валидации: {-scores.mean()}")

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Среднеквадратичная ошибка с RandomForest: {mse}")

# Интерфейс Streamlit
st.title("Прогнозирование стоимости недвижимости")


# Поля ввода для признаков
total_square = st.number_input("Общая площадь (м²):", min_value=0.0, step=0.1)
rooms = st.number_input("Количество комнат:", min_value=0, step=1)

# Кнопка для выполнения предсказания
if st.button("Рассчитать стоимость"):
    if total_square > 0 and rooms > 0:
        # Используем pandas DataFrame для корректного предсказания
        prediction_input = pd.DataFrame([[total_square, rooms]], columns=["total_square", "rooms"])
        prediction = model.predict(prediction_input)
        st.write(f"Предполагаемая стоимость: {prediction[0]:,.2f} руб.")
    else:
        st.write("Заполните все поля.")