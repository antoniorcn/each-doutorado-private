# salvar_modelo.py
from typing import Tuple
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D


def processar_imagem(image_path : str, tamanho: Tuple[int, int]=(244, 244)) -> np.ndarray:
    """
    Função para carregar e processar a imagem, retornando
    um numpy array no formato (244, 244, 3) normalizado
    """
    # logger.debug(f"Lendo imagem de: {image_path}")
    imagem = Image.open(image_path)

    # Garantir que a imagem tenha 3 canais (RGB)
    imagem = imagem.convert("RGB")

    # Redimensionar a imagem para 244x244
    imagem = imagem.resize(tamanho)

    # Converter a imagem em um array numpy
    pixels = np.array(imagem, dtype=np.float32)

    # Normalizar os pixels para o intervalo [0, 1]
    pixels /= 255.0
    return pixels

model = Sequential([
    Input(shape=(244, 244, 3)),
    Conv2D(filters=1, kernel_size=3, activation='relu', name="conv2d_1")
])

model.save("modelo_tf.keras")
