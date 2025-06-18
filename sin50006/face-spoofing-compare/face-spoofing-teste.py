import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from custom_model_controller import CustomModelController
from deepface_controller import DeepFaceRecognitionController
import numpy as np
import random
import tensorflow as tf
from typing import List, Optional, TypedDict, Callable, Tuple, Union
from logger import get_logger_arquivo
from processador_image import carregar_imagens_com_label, processar_imagem


SPOOF_DADOS_PATH = "c:\\git\\dados\\sin50006\\casia-fasd"
SPOOF_CASIA_FASD_PATH = "c:\\git\\dados\\sin50006\\casia-fasd\\test"

class FaceClassSubSet(TypedDict):
    relative_path : str
    prefix : str
    label : str
    instance_start : int
    instance_end: int
    version_start: int
    version_end: int
    version_prefix: str


def load_files_names(subsets : List[FaceClassSubSet]) -> List[Tuple[str, int]]:
    image_paths : List[Tuple[str, int]] = []
    for subset in subsets:
        logger.debug(f"Carregado subset de imagens de: {subset['relative_path']}{subset['prefix']}")
        logger.debug(f"Carregado imagens de: {subset['instance_start']} ate: {subset['instance_end']}")
        image_start_path : List[str] = []
        for instance_index in range(subset["instance_start"], subset["instance_end"]):
            logger.debug(f"Carregado imagem: {instance_index}")
            for version_index in range(subset["version_start"], subset["version_end"]):
                wildcard = f"{subset['relative_path']}{subset['prefix']}{instance_index}"
                wildcard += f"{subset['version_prefix']}{version_index}*.*"
                relative_file_path = os.path.join(SPOOF_CASIA_FASD_PATH, wildcard)
                image_start_path.append(relative_file_path)
                logger.debug(f"Adicionando imagens de: {relative_file_path} ao pacote de carga")
        images_paths, image_labels = carregar_imagens_com_label(image_start_path, subset["label"], None)
        image_paths.extend(
                list(zip( images_paths, image_labels ))
        )
    return image_paths


def load_test_validation_samples():
    subsets = [
        {"relative_path": "live\\",
         "prefix": "s", "label": 0, "instance_start": 1, "instance_end": 30,
         "version_start": 1, "version_end": 2, "version_prefix": "v"},
        {"relative_path": "spoof\\",
         "prefix": "s", "label": 1, "instance_start": 1, "instance_end": 30,
         "version_start": 3, "version_end": 8, "version_prefix": "v"},
        {"relative_path": "spoof\\",
         "prefix": "s", "label": 1, "instance_start": 1, "instance_end": 30,
         "version_start": 1, "version_end": 4, "version_prefix": "vHR_"}
    ]
    logger.info("Carregando imagens de testes e validacao")
    tests_validation_files = load_files_names(subsets)
    logger.info(f"Foram identificadas: {len(tests_validation_files)} imagens de testes e treinamento")
    return tests_validation_files


logger = get_logger_arquivo(__name__)

logger.info(f"SPOOF_DADOS_PATH: {SPOOF_DADOS_PATH}")
logger.info(f"SPOOF_CASIA_FASD_PATH: {SPOOF_CASIA_FASD_PATH}")

if __name__ == "__main__":
    deepface_controller = DeepFaceRecognitionController()
    custom_model_controller = CustomModelController()
    validation_files = load_test_validation_samples()
    custom_model_fp = 0
    custom_model_fn = 0
    custom_model_tp = 0
    custom_model_tn = 0
    custom_model_acerto = 0
    deepface_fp = 0
    deepface_fn = 0
    deepface_tp = 0
    deepface_tn = 0
    deepface_acerto = 0
    image_index = 0

    for test_image_file in validation_files[0:10]:
        image_filename, image_label =  test_image_file
        image, label = processar_imagem(image_filename, image_label)
        logger.debug(f"Image Shape: {image.shape}  Label : {label}")
        spoofing = label == 1

        input_data = image.astype('float32') / 255.0
        input_data = np.expand_dims(input_data, axis=0)  # (1, 244, 244, 3)
        custom_model_realface = custom_model_controller.is_real_face(input_data)
        deepface_realface = 0 if not deepface_controller.is_real_face( image ) else 1

        # 0 - Real Face     1 - Spoofing
        if custom_model_realface and not spoofing:
            custom_model_tn += 1
            custom_model_acerto += 1
        elif custom_model_realface and spoofing:
            custom_model_fn += 1
        elif not custom_model_realface and spoofing:
            custom_model_tp += 1
            custom_model_acerto += 1
        elif not custom_model_realface and not spoofing:
            custom_model_fp += 1

        if deepface_realface and not spoofing:
             deepface_tn += 1
             deepface_acerto += 1
        elif deepface_realface and spoofing:
             deepface_fn += 1
        elif not deepface_realface and spoofing:
             deepface_tp += 1
             deepface_acerto += 1
        elif not deepface_realface and not spoofing:
             deepface_fp += 1
        # image_normalizada = tf.cast(img_array, tf.float32) / 255.0  # normaliza
        texto = f"Image Index: {image_index}    Custom Acertos: {custom_model_acerto}     Deep Face Acertos: {deepface_acerto}"
        logger.info(texto)
    
    logger.info(f"Custom Model - Falso Positivos: {custom_model_fp}")
    logger.info(f"Custom Model - Falso Negativos: {custom_model_fn}")
    logger.info(f"Custom Model - True Positivos: {custom_model_tp}")
    logger.info(f"Custom Model - True Negativos: {custom_model_tn}")
    logger.info(f"Custom Model - Acertos: {custom_model_acerto}")
    logger.info(f"Deep Face - Falso Positos: {deepface_fp}")
    logger.info(f"Deep Face - Falso Negativos: {deepface_fn}")
    logger.info(f"Deep Face - True Positivos: {deepface_tp}")
    logger.info(f"Deep Face - True Negativos: {deepface_tn}")
    logger.info(f"Deep Face - Acertos: {deepface_acerto}")
