# Weather AI Dashboard

Проєкт завантажує дані з датасету **Weather Type Classification**, порівнює кілька методів машинного навчання та демонструє роботу найкращої моделі в інтерфейсі Streamlit:

- прогноз погоди;
- ймовірності всіх класів;
- таблиця й графіки вихідних даних;
- порівняння моделей;
- матриця помилок і крива навчання.

У проєкті використовуються `scikit-learn`, `pandas`, `Plotly` і `Streamlit`.

## Структура проєкту

```
weather_ai_dashboard/
├── data/
│   └── weather_classification_data.csv
├── artifacts/
├── train.py
├── predictor.py
├── app.py
├── test_predictor.py
└── requirements.txt
```

## Запуск у Windows

Відкрийте термінал у папці проєкту та виконайте команди:

```powershell
cd d:\repos\FimalProjectAi
python -m venv .venv
.venv\Scripts\Activate.ps1
cd weather_ai_dashboard
python -m pip install -r requirements.txt
```

Якщо датасету ще немає, додайте файл `weather_classification_data.csv` до папки `weather_ai_dashboard/data/` вручну або завантажте його з Kaggle.

Потім навчіть модель:

```powershell
python train.py
```

Після успішного навчання запустіть вебінтерфейс:

```powershell
streamlit run app.py
```

Відкрийте адресу, яка з’явиться в терміналі, зазвичай:

```text
http://localhost:8501
```

## Тестування

У папці `weather_ai_dashboard` виконайте:

```powershell
python -m pytest -q
```

## Як працює проєкт

Скрипт `train.py` завантажує дані, перевіряє ознаки та пропуски, розділяє вибірку на навчальну й тестову, навчає базову модель і кілька основних моделей. Для `MLPClassifier` підбирання параметрів здійснюється за допомогою `GridSearchCV`.

Найкращий pipeline зберігається в папку `artifacts/`. Модуль `predictor.py` перевіряє вхідні дані та виконує прогноз, а `app.py` відображає результат і графіки у Streamlit.
