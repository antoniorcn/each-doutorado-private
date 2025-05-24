"""Módulo para leitura e processamento inicial de imagens utilizando Pillow"""

from typing import Tuple, List
from PIL import Image
import numpy as np
import os
from glob import glob
from random import sample
from spoofing.edu.ic.logger import get_logger_arquivo
import tensorflow as tf

logger = get_logger_arquivo(__name__)


# def processar_imagem(image_path : str, label : str = None, tamanho: Tuple[int, int]=(244, 244)) -> np.ndarray:
#     """
#     Função para carregar e processar a imagem, retornando
#     um numpy array no formato (244, 244, 3) normalizado
#     """
#     # logger.debug(f"Lendo imagem de: {image_path}")
#     imagem = Image.open(image_path)

#     # Garantir que a imagem tenha 3 canais (RGB)
#     imagem = imagem.convert("RGB")

#     # Redimensionar a imagem para 244x244
#     imagem = imagem.resize(tamanho)

#     # Converter a imagem em um array numpy
#     pixels = np.array(imagem, dtype=np.float32)

#     # Normalizar os pixels para o intervalo [0, 1]
#     pixels /= 255.0
#     return pixels, label


# def processar_imagem_caminho(image_path : str, tamanho: Tuple[int, int]=(244, 244), samples=10) -> list:
#     image_paths = glob(image_path)
#     image_files = [
#         f for f in image_paths
#         if f.lower().endswith(('.png', '.jpg', '.jpeg'))
#     ]
#     selected_files = sample(image_files, min(samples, len(image_files)))
#     images : list = []
#     for filename in selected_files:
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#             img_path = os.path.join(image_path, filename)
#             img_array, _ = processar_imagem(image_path=img_path, tamanho=tamanho)
#             images.append(img_array)
#     return images


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
        logger.info("carregando imagens de " + image_path)
        image_files.extend(glob(image_path))

    image_files.sort()  # Ordena para garantir consistência
    if samples is not None:
        image_files = image_files[:samples]

    labels = [label] * len(image_files)

    return image_files, labels


def processar_imagem_tf(image_path, label, tamanho=(224, 224)):
    image = tf.io.read_file(image_path)  # lê o conteúdo do arquivo
    # Detecta a extensão
    is_jpg = tf.strings.regex_full_match(image_path, ".*\\.jpe?g")

    # Condicional de decodificação
    if is_jpg:
        image = tf.image.decode_jpeg(image, channels=3)
    else:
        image = tf.image.decode_png(image, channels=3)
    logger.debug(f"Imagem: {image}  decode usado {'JPG' if is_jpg else 'PNG'} redimensionando para o tamanho {tamanho}")
    image = tf.image.resize(image, tamanho)  # redimensiona
    logger.debug("Imagem redimensionada")
    image = tf.cast(image, tf.float32) / 255.0  # normaliza
    logger.debug("Imagem normalizada")
    logger.debug(f"Retornando Imagem: {image}   Label: {label}")
    return image, label


def criar_dataset(
    image_paths_labels : List[Tuple[str, int]],
    tamanho: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    shuffle: bool = False
):
    caminhos, labels = zip(*image_paths_labels)

    caminhos = list(caminhos)
    labels = list(labels)

    ds = tf.data.Dataset.from_tensor_slices((caminhos, labels))
    logger.debug("dataset criado")

    if shuffle:
        ds = ds.shuffle(buffer_size=len(caminhos))
        logger.debug("dataset embaralhado")
        
    # Aplica o pré-processamento
    logger.debug(f"Dataset possui: {sum(dataset_length(ds)[1])} nome de imagens de arquivos")
    ds = ds.map(lambda x, y: processar_imagem_tf(x, y, tamanho), num_parallel_calls=tf.data.AUTOTUNE)
    logger.debug(f"Foram carregadas: {dataset_length(ds)} images destes arquivos")

    # Agrupa em batches
    ds = ds.batch(batch_size)
    logger.debug("dataset agrupado em batches")

    # Prefetch para performance
    ds = ds.prefetch(tf.data.AUTOTUNE)
    logger.debug("aplicado prefetch para performance")

    return ds



# def criar_dataset(
#     image_paths : List[str],
#     label,
#     samples: int = None,
#     tamanho: Tuple[int, int] = (224, 224),
#     batch_size: int = 32,
#     shuffle: bool = False
# ):
#     logger.debug(f"Acessando pasta(s): {image_paths}")
#     caminhos, labels = carregar_imagens_com_label(image_paths, label=label, samples=samples)
#     # Cria o dataset base
#     logger.debug(f"Lendo imagem de: {caminhos} e labels {labels}")
#     ds = tf.data.Dataset.from_tensor_slices((caminhos, labels))
#     logger.debug("dataset criado")
#     # Limita a quantidade de amostras (se solicitado)
#     if samples is not None:
#         ds = ds.take(samples)
#         logger.debug("samples escolhidos")
#     # Embaralhamento (antes do mapeamento)
#     if shuffle:
#         ds = ds.shuffle(buffer_size=len(caminhos))
#         logger.debug("dataset embaralhado")
#     # Aplica o pré-processamento
#     logger.debug(f"Dataset possui: {sum(dataset_length(ds)[1])} nome de imagens de arquivos")
#     ds = ds.map(lambda x, y: processar_imagem_tf(x, y, tamanho), num_parallel_calls=tf.data.AUTOTUNE)
#     logger.debug(f"Foram carregadas: {dataset_length(ds)} images destes arquivos")

#     # Agrupa em batches
#     ds = ds.batch(batch_size)
#     logger.debug("dataset agrupado em batches")

#     # Prefetch para performance
#     ds = ds.prefetch(tf.data.AUTOTUNE)
#     logger.debug("aplicado prefetch para performance")

#     return ds


def calcular_tamanho_bytes(tensor: tf.Tensor) -> tf.Tensor:
    logger.debug(f"Calculando tamanho em bytes da image {tensor}")
    # Mapeia tipos para tamanho em bytes
    bytes_por_tipo = {
        tf.float32: 4,
        tf.float64: 8,
        tf.int32: 4,
        tf.int64: 8,
    }

    if tensor.dtype not in bytes_por_tipo:
        logger.error(f"Tipo de dado não suportado: {tensor.dtype}")

    # Calcula número total de elementos
    num_elementos = tf.size(tensor)  # total de valores no tensor

    # Tamanho por elemento
    bytes_por_elemento = bytes_por_tipo[tensor.dtype]

    return num_elementos * bytes_por_elemento

# def dataset_length(dataset):
#     logger.debug(f"Calculando quantidade de elementos no dataset {dataset}")
#     total = 0
#     for x, _ in dataset:
#         logger.debug(f"Shape: {tf.shape(x)} Tipo: {type(tf.shape(x))}")
#         if type(tf.shape(x)) is EagerTensor: 
#             batch_size = tf.shape(x)[0].numpy()
#         else:
#             batch_size = 1
#         total += batch_size
#     logger.debug(f"Total de elementos no dataset {total}")
#     return total



def dataset_length(dataset, campo=0):
    num_batches = 0
    elementos_por_batch = []

    for batch in dataset:
        num_batches += 1

        # Determina o elemento a ser avaliado
        if isinstance(batch, (tuple, list)):
            x = batch[campo]
        elif isinstance(batch, dict):
            if isinstance(campo, str):
                x = batch[campo]
            else:
                logger.error("Para batches do tipo dict, 'campo' deve ser uma string com a chave.")
        else:
            x = batch  # Tensor direto

        # Valida que x é tensor com dimensão válida
        if isinstance(x, tf.Tensor):
            if x.shape.rank == 0:
                elementos_por_batch.append(1)
            else:
                elementos_por_batch.append(x.shape[0])
        else:
            logger.warning(f"Tipo inesperado no batch {num_batches}: {type(x)}")
            elementos_por_batch.append(0)

    return num_batches, elementos_por_batch