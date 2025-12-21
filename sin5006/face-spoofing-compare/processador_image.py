"""Módulo para leitura e processamento inicial de imagens utilizando Pillow"""

import os
from typing import Tuple, List
from PIL import Image
import numpy as np
from glob import glob
from logger import get_logger_arquivo

logger = get_logger_arquivo(__name__)


def carregar_imagens_com_label(
    image_paths : List[str],
    label,
    samples: int = None
) -> Tuple[List[str], List[int]]:
    """
    Lê arquivos de imagem do diretório especificado e associa o mesmo label a todos.

    :param image_path: Caminho da pasta com as imagens
    :param label: Label que será associado a todas as imagens
    :param samples: Quantidade máxima de arquivos a carregar
    :param extensoes_permitidas: Extensões válidas (ex: jpg, png)
    :return: (lista_de_caminhos, lista_de_labels)
    """

    image_files = []
    for image_path in image_paths:
        logger.debug("carregando imagens de " + image_path)
        image_files.extend(glob(image_path))

    image_files.sort()  # Ordena para garantir consistência
    if samples is not None:
        image_files = image_files[:samples]

    labels = [label] * len(image_files)

    return image_files, labels


def processar_imagem(image_path, label, tamanho=(244, 244)):
    image = Image.open(image_path)  # lê o conteúdo do arquivo
    # Detecta a extensão
    image = image.convert('RGB')
    image = image.resize(tamanho)
    img_array = np.array(image)
    return img_array, label

