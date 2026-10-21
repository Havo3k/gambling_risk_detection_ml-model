# gambling_risk_detection_ml-model
Interpretable ML model for early detection of gambling-addiction risk from bank transaction patterns (synthetic data)
Прототип интерпретируемой ML-модели (логистическая регрессия), которая по
транзакционным паттернам клиентов банка оценивает риск формирования игровой
зависимости. Данные синтетические; реальные данные клиентов не используются.

Содержание:
- generate_dataset.py — генерация синтетического набора транзакций
- train_model.py — расчёт 8 поведенческих показателей, обучение модели, метрики
- feature_samples.csv, transaction_samples.csv — примеры данных

Инструкция к запуску:
1. pip install -r requirements.txt
2. python generate_dataset.py
3. python train_model.py

Результаты:
Доля верных ответов 93,6%, ROC-AUC 0,905.
(Результаты получены на синтетических данных и требуют проверки на реальных.)

Ограничения:
Прототип учебный: модель не предназначена для автоматического принятия решений
о клиентах без участия человека.
