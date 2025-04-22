"""Módulo para leitura e processamento inicial de imagens utilizando Pillow"""

from typing import Tuple
from PIL import Image
import numpy as np
import os
from random import sample
from spoofing.edu.ic.logger import get_logger

logger = get_logger(__name__)


def processar_imagem(image_path : str, tamanho: Tuple[int, int]=(244, 244)) -> np.ndarray:
    """
    Função para carregar e processar a imagem, retornando
    um numpy array no formato (244, 244, 3) normalizado
    """
    logger.debug(f"Lendo image de: {image_path}")
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


def processar_imagem_caminho(image_path : str, tamanho: Tuple[int, int]=(244, 244), samples=10) -> list:
    image_files = [
        f for f in os.listdir(image_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    selected_files = sample(image_files, min(samples, len(image_files)))
    images : list = []
    for filename in selected_files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_path, filename)
            img_array = processar_imagem(image_path=img_path, tamanho=tamanho)
            images.append(img_array)
    return images
