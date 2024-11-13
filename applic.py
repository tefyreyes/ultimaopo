import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Carga de tu modelo de reconocimiento de frutas
model = tf.keras.models.load_model('modelo_entrenado.keras')

# Etiquetas de las clases de frutas (ajusta según tu modelo)
etiquetas = ['Banana', 'Berenjena', 'Cebolla', 'Cereza', 'Choclo', 'Cocos', 'Coliflor', 'Frutilla', 
             'Kiwi', 'Limon', 'Manzana', 'Morron', 'Naranja', 'Palta', 'Papa', 'Pepino', 'Pera', 
             'Piña', 'Repollo', 'Sandia', 'Tomate', 'Uva', 'Zanahoria', 'Zucchini']

st.title('Clasificador de Frutas')

# Cargar imagen
uploaded_file = st.file_uploader("Carga una imagen de una fruta", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen cargada', use_container_width=True)

    # Asegurarse de que la imagen tenga 3 canales (RGB)

    image = image.convert("RGB")  # Convertir la imagen a RGB si tiene un canal alfa

    # Preprocesamiento de la imagen
    img_array = np.array(image.resize((100, 100)))  # Ajuste del tamaño (100x100 según tu modelo)
    img_array = img_array / 255.0  # Normalizar la imagen
    img_array = np.expand_dims(img_array, axis=0)  # Añadir la dimensión de batch

    # Realizar la predicción
    predictions = model.predict(img_array)
    indice_prediccion = np.argmax(predictions)  # Obtener el índice de la clase con mayor probabilidad
    fruta_predicha = etiquetas[indice_prediccion]  # Obtener la etiqueta correspondiente

    # Mostrar la predicción y la confianza
    st.write("Esta fruta es probablemente:", fruta_predicha)
    st.write(f"Confianza: {round(predictions[0][indice_prediccion] * 100, 2)} %")