# kaggle-house-prices-prediction
Предсказание стоимости недвижимости. Проект включает разведочный анализ данных (EDA) и построение прогнозной модели с использованием XGBoost Regressor на основе открытого датасета Kaggle - https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques.

## Ключевые этапы
1.  **Разведочный анализ (EDA)**: обработка пропусков и выбросов, анализ целевой переменной + **Статистический анализ**: проверка гипотез, сравнение групп по категориальным признакам, корреляционный анализ.
2.  **Feature Engineering**: преобразование и создание новых признаков для улучшения модели.
3.  **Моделирование**: обучение и настройка XGBoost Regressor.

## Результат модели
Финальная модель была оценена метрикой **RMSLE** (Root Mean Squared Logarithmic Error), соответствующей условиям соревнования Kaggle.

```python
predictions = model_2.predict(X_test_2)
RMSLE = np.sqrt(mean_squared_log_error(y_test, predictions))
print(f"The score is {RMSLE:.5f}")

The score is 0.13096
