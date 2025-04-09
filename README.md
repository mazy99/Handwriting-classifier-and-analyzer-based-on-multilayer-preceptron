
# Handwriting Classifier and Analyzer Based on Multilayer Perceptron

## Описание проекта
Этот проект представляет собой классификатор и анализатор почерка, основанный на многослойном перцептроне (MLP). Он использует современные технологии машинного обучения и глубокого обучения для анализа и классификации рукописных текстов.

## Используемый стек технологий
- **TensorFlow** + **Keras** – обучение многослойного перцептрона (MLP);
- **OpenCV (cv2)** – предобработка изображений;
- **PIL (Pillow)** – работа с изображениями;
- **FastAPI** – реализация HTTP-сервера для взаимодействия с моделью;
- **Matplotlib** + **Seaborn** – визуализация данных и результатов;
- **NumPy** – обработка числовых данных;
- **Pandas** – анализ и обработка табличных данных.

## Архитектура модели
Модель построена на многослойном перцептроне с функцией активации **ReLU** (Rectified Linear Unit). Она позволяет эффективно обучаться на больших объемах данных и избегает проблемы исчезающего градиента.



<pre>├── data/
│   ├── raw/              # исходные картинки текста
│   └── processed/        # обработанные изображения
│
├── src/
│   ├── preprocessing.py  # функции для обработки картинок
│   ├── features.py       # функции для извлечения признаков
│   └── model.py          # машинное обучение (если потребуется)
│
├── requirements.txt
├── README.md
└── .gitignore </pre>

## HTTP API
Сервер на основе **FastAPI** обрабатывает HTTP-запросы и возвращает соответствующие статусы:

### Эндпоинты
#### 1. `POST /predict` – классификация почерка
```python
from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = FastAPI()
model = load_model("model.h5")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('L')
        image = image.resize((28, 28))
        image = np.array(image) / 255.0
        image = image.reshape(1, 28*28)
        prediction = model.predict(image)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}
```
- **Запрос**: изображение в формате `multipart/form-data`
- **Ответ**: JSON с предсказанным классом и вероятностями
- **Статусы**:
  - `200 OK` – успешное предсказание
  - `400 Bad Request` – неверный формат данных
  - `500 Internal Server Error` – внутренняя ошибка сервера

#### 2. `GET /health` – проверка состояния сервера
```python
@app.get("/health")
def health_check():
    return {"status": "ok"}
```
- **Ответ**: JSON с информацией о состоянии сервера
- **Статусы**:
  - `200 OK` – сервер работает корректно
  - `500 Internal Server Error` – ошибка в работе сервера

## Установка и запуск
### Требования
- Python 3.8+
- Установленные зависимости (см. `requirements.txt`)

### Установка зависимостей
```sh
pip install -r requirements.txt
```

### Запуск сервера
```sh
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Авторы
Разработчики и участники проекта. Если у вас есть вопросы или предложения, обращайтесь в Issues.

---
Этот README файл содержит всю основную информацию о проекте, его архитектуре и API.

