from PIL import Image
import numpy as np


def processar_imagem(image_path : str):
    """
    Função para carregar e processar a imagem, retornando
    um numpy array no formato (244, 244, 3) normalizado
    """
    imagem = Image.open(image_path)
    
    # Garantir que a imagem tenha 3 canais (RGB)
    imagem = imagem.convert("RGB")

    # Redimensionar a imagem para 244x244
    imagem = imagem.resize((244, 244))

    # Converter a imagem em um array numpy
    pixels = np.array(imagem, dtype=np.float32)

    # Normalizar os pixels para o intervalo [0, 1]
    pixels /= 255.0
    return pixels
