import datetime
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from predictor import WeatherPredictor

st.set_page_config(page_title="Анализ погоды", page_icon="🌦️", layout="wide")

WEATHER_NAMES = {
    "Sunny": "Ясно",
    "Cloudy": "Облачно",
    "Rainy": "Дождливо",
    "Snowy": "Снег",
}


@st.cache_resource
def load_predictor():
    return WeatherPredictor("artifacts")


@st.cache_data
def load_artifacts():
    with open("artifacts/metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    eda_df = pd.read_csv("artifacts/eda_data.csv")
    return metrics, eda_df


if not Path("artifacts/model.joblib").exists():
    st.error("Сначала нужно обучить модель командой: python train.py")
    st.stop()

predictor = load_predictor()
metrics, eda_df = load_artifacts()
bounds = predictor.bounds

st.title("Метеорологический анализ погоды")
st.caption("Практическая работа по машинному обучению")

st.sidebar.header("Параметры для расчёта")
st.sidebar.write("Измените значения и посмотрите, какой тип погоды будет определён.")

allow_out_of_bounds = st.sidebar.checkbox(
    "Проверить значения вне диапазона датасета",
    value=False,
)

user_input = {}
for col, stats in bounds["numeric"].items():
    min_value = float(stats["min"])
    max_value = float(stats["max"])

    if allow_out_of_bounds:
        span = max_value - min_value
        min_value -= span * 0.3
        max_value += span * 0.3

    user_input[col] = st.sidebar.slider(
        col,
        min_value=round(min_value, 1),
        max_value=round(max_value, 1),
        value=round(float(stats["median"]), 1),
        step=0.5,
    )

for col, options in bounds["categorical"].items():
    user_input[col] = st.sidebar.selectbox(col, options)


tab_forecast, tab_eda, tab_results = st.tabs([
    "Прогноз",
    "Данные и графики",
    "Модели и ошибки",
])

with tab_forecast:
    forecast_col1, forecast_col2 = st.columns([1, 3])
    with forecast_col1:
        forecast_days = st.slider("Горизонт прогноза, дней", 1, 7, 3)
        start_date = st.date_input("Начальная дата", value=datetime.date.today())

    st.caption(
        "Прогноз рассчитывается обученной классификационной моделью с плавным изменением погодных показателей по датам."
    )

    forecast_results = []
    for offset in range(forecast_days):
        forecast_date = start_date + datetime.timedelta(days=offset)
        forecast_input = dict(user_input)

        seasonal_additions = {
            "Temperature": (2.2 * (offset % 3) - 1.4) * (1 if offset < 5 else -1),
            "Humidity": 5 * (1 if offset % 2 == 0 else -1),
            "Wind Speed": 1.8 * ((offset % 3) - 1),
            "Precipitation (%)": 6 * (1 if offset % 2 else -0.4),
            "Atmospheric Pressure": -1.2 * (offset % 4),
            "UV Index": 0.8 * (1 if offset % 3 == 0 else -0.3),
            "Visibility (km)": -1.2 * (offset % 2),
        }

        for feature, addition in seasonal_additions.items():
            if feature not in forecast_input:
                continue
            stats = bounds["numeric"][feature]
            adjusted = float(forecast_input[feature]) + addition
            forecast_input[feature] = round(min(max(adjusted, stats["min"]), stats["max"]), 1)

        month = forecast_date.month
        if "Season" in bounds["categorical"]:
            english_season = (
                "Winter" if month in (12, 1, 2)
                else "Spring" if month in (3, 4, 5)
                else "Summer" if month in (6, 7, 8)
                else "Autumn"
            )
            if english_season in bounds["categorical"]["Season"]:
                forecast_input["Season"] = english_season

        try:
            result = predictor.validate_and_predict(forecast_input)
            forecast_results.append({
                "date": forecast_date,
                "input": forecast_input,
                "result": result,
            })
        except ValueError as error:
            st.error(f"Проверьте введённые данные: {error}")

    if forecast_results:
        day_columns = st.columns(len(forecast_results))
        day_labels = []
        for index, item in enumerate(forecast_results):
            result = item["result"]
            weather = WEATHER_NAMES.get(result["prediction"], result["prediction"])
            date_label = item["date"].strftime("%d.%m")
            day_labels.append(date_label)
            with day_columns[index]:
                st.metric(date_label, weather)
                st.caption(f"Уверенность: {result['confidence'] * 100:.1f}%")

        selected_date = st.selectbox(
            "Подробнее о дате",
            options=day_labels,
            index=0,
        )
        selected_index = day_labels.index(selected_date)
        selected = forecast_results[selected_index]
        result = selected["result"]

        for warning in result["warnings"]:
            st.warning(warning)

        result_col, chart_col = st.columns([1, 1.5])

        with result_col:
            st.subheader(f"Результат на {selected['date'].strftime('%d.%m.%Y')}")
            weather = WEATHER_NAMES.get(result["prediction"], result["prediction"])
            st.metric("Тип погоды", weather)
            st.metric("Доля уверенности", f"{result['confidence'] * 100:.1f}%")

            st.markdown("**Наиболее значимые признаки:**")
            for feature, importance in result["top_features"]:
                st.write(f"- {feature}: {importance * 100:.1f}%")

        with chart_col:
            st.subheader("Вероятности типов погоды")
            probability_df = pd.DataFrame({
                "Тип погоды": [
                    WEATHER_NAMES.get(name, name)
                    for name in result["probabilities"]
                ],
                "Вероятность, %": [
                    value * 100
                    for value in result["probabilities"].values()
                ],
            })

            probability_figure = px.bar(
                probability_df,
                x="Вероятность, %",
                y="Тип погоды",
                orientation="h",
                text="Вероятность, %",
                color="Тип погоды",
                range_x=[0, 100],
            )
            probability_figure.update_layout(showlegend=False, height=320)
            st.plotly_chart(probability_figure, use_container_width=True)

        st.subheader("Параметры выбранного дня")
        st.dataframe(pd.DataFrame([selected["input"]]), use_container_width=True, hide_index=True)

with tab_eda:
    dataset_info = metrics["dataset_info"]

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Всего записей", f"{dataset_info['total_rows']:,}")
    metric_col2.metric("Пропусков", dataset_info["missing_values"])
    metric_col3.metric("Типов погоды", len(dataset_info["class_distribution"]))

    st.subheader("Таблица данных")
    selected_weather = st.multiselect(
        "Тип погоды",
        options=bounds["classes"],
        default=bounds["classes"],
    )
    filtered_df = eda_df[eda_df["Weather Type"].isin(selected_weather)]
    st.dataframe(filtered_df, use_container_width=True, height=300)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Температура")
        temperature_figure = px.histogram(
            filtered_df,
            x="Temperature",
            color="Weather Type",
            barmode="overlay",
            nbins=40,
            opacity=0.7,
        )
        st.plotly_chart(temperature_figure, use_container_width=True)

    with chart_col2:
        st.subheader("Влажность и осадки")
        sample_df = filtered_df.sample(min(600, len(filtered_df)), random_state=42)
        precipitation_figure = px.scatter(
            sample_df,
            x="Humidity",
            y="Precipitation (%)",
            color="Weather Type",
            opacity=0.75,
        )
        st.plotly_chart(precipitation_figure, use_container_width=True)

    st.subheader("Количество наблюдений")
    class_df = pd.DataFrame({
        "Тип погоды": list(dataset_info["class_distribution"]),
        "Количество": list(dataset_info["class_distribution"].values()),
    })
    class_figure = px.bar(
        class_df,
        x="Тип погоды",
        y="Количество",
        color="Тип погоды",
    )
    st.plotly_chart(class_figure, use_container_width=True)

with tab_results:
    st.subheader("Сравнение методов")
    comparison_df = pd.DataFrame(metrics["comparison"])
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    matrix_col, learning_col = st.columns(2)

    with matrix_col:
        st.subheader("Матрица ошибок")
        cm_data = metrics["confusion_matrix"]
        confusion_figure = px.imshow(
            cm_data["matrix"],
            x=cm_data["labels"],
            y=cm_data["labels"],
            text_auto=True,
            color_continuous_scale="Blues",
            labels={"x": "Предсказанный класс", "y": "Истинный класс"},
        )
        st.plotly_chart(confusion_figure, use_container_width=True)

    with learning_col:
        st.subheader("Точность при разном объёме данных")
        learning_df = pd.DataFrame(metrics["learning_curve"])
        learning_figure = px.line(
            learning_df,
            x="Записей в обучении",
            y="Test Accuracy",
            markers=True,
        )
        st.plotly_chart(learning_figure, use_container_width=True)

    st.subheader("Частые ошибки классификации")
    st.dataframe(
        pd.DataFrame(metrics["top_errors"]),
        use_container_width=True,
        hide_index=True,
    )
