import streamlit as st
import joblib
import re
import string
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

# Загрузка сохраненных модели и векторизатора
try:
    pac_model = joblib.load('spam_classifier_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    st.error("Ошибка: Файлы модели (spam_classifier_model.pkl) или векторизатора (tfidf_vectorizer.pkl) не найдены.")
    st.warning("Пожалуйста, убедитесь, что вы загрузили их в корневую директорию Colab.")
    st.stop()

# Функция для предобработки текста (такая же, как при обучении)
def clean_text(text):
    if text.startswith('Subject: '):
        text = text[text.find(' ')+1:]

    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = ''.join([char for char in text if char not in string.punctuation])
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Интерфейс Streamlit-приложения
st.set_page_config(page_title="Система определения спам-писем")

st.title("Система определения спам-писем")
st.markdown("---")
st.write("Введите текст электронного письма ниже, чтобы определить, является ли оно спамом.")

# Поле для ввода текста пользователем
email_text = st.text_area("Текст письма", height=200, placeholder="Введите текст письма здесь...")

# Кнопка для запуска предсказания
if st.button("Проверить на спам"):
    if email_text:
        with st.spinner("Анализируем письмо..."):
            # Предобработка введенного текста
            cleaned_input_text = clean_text(email_text)

            # Векторизация текста
            vectorized_input = tfidf_vectorizer.transform([cleaned_input_text])

            # Предсказание моделью
            prediction = pac_model.predict(vectorized_input)

        st.markdown("---")
        st.subheader("Результат:")

        if prediction[0] == 1:
            st.error("**ВНИМАНИЕ! Это письмо, вероятно, является СПАМОМ!**")
        else:
            st.success("**Это письмо, скорее всего, НЕ является спамом.**")
    else:
        st.warning("Пожалуйста, введите текст письма для анализа.")

st.markdown("---")
st.info("Это демонстрационный сервис. Результаты модели могут не быть 100% точными.")