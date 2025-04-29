from typing import List, Optional, TypedDict, Callable, Tuple, Union
from enum import IntEnum
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D,\
    Dense, Dropout, Add, Activation
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import Callback
from spoofing.edu.ic.processador_image import processar_imagem_caminho
from spoofing.edu.ic.dropblock2d import DropBlock2D
from spoofing.edu.ic.logger import get_logger_arquivo
from spoofing.edu.ic.metricas_adicionais import EERHTERCallback

logger = get_logger_arquivo(__name__)
logger.info("Detector inicializado")

def get_env_var(name: str, default: str = None) -> str:
    """Lê uma variável de ambiente com nome `name`.
       Se não existir, retorna o `default`.
    """
    value = os.getenv(name, default)
    return value

SPOOF_DADOS_PATH = get_env_var("SPOOF_DADOS_PATH", default="C:\\git\\dados\\sin50006")
SPOOF_CASIA_FASD_PATH = os.path.join(SPOOF_DADOS_PATH, "casia-fasd")
SPOOF_MODEL_PATH = os.path.join(SPOOF_DADOS_PATH, "modelos")

# ## Apenas Teste 
# file_name = f"model-samples-{10}" +\
#                 f"-epochs-{10}.weights.h5"
# MODEL_FILE = os.path.join(SPOOF_MODEL_PATH, file_name)
# logger.info(f"MODEL FILE: {MODEL_FILE}")
# ## Apenas Teste 

logger.info(f"SPOOF_DADOS_PATH: {SPOOF_DADOS_PATH}")
logger.info(f"SPOOF_CASIA_FASD_PATH: {SPOOF_CASIA_FASD_PATH}")
logger.info(f"SPOOF_MODEL_PATH: {SPOOF_MODEL_PATH}")


class FaceClassSubSet(TypedDict):
    relative_path : str 
    prefix : str
    label : str
    instance_start : int
    instance_end: int
    version_start: int
    version_end: int
    version_prefix: str

# Definindo uma Enumeração
class SpoofingEstados(IntEnum):
    """Enumeration para controlar o estado da classe Spoofing"""
    NEW = 1
    GENERATED = 2
    COMPILED = 3
    TRAINED = 4
    EVALUATED = 5
    READY = 6

class SpoofingException(Exception):
    """Classe de Excepção para Spoofing"""
    def __init__(self, mensagem):
        super().__init__(mensagem)
        self.mensagem = mensagem


class Spoofing:
    """Classe Spoofing que gera, compila, treina e salva os pesos gerados"""
    def __init__(self, optimizer="adam", loss='binary_crossentropy', metrics='accuracy'):
        self.kernel_size : Tuple[int, int] = (7, 7)
        self.strides : Tuple[int, int] = (2, 2)
        self.filters : int = 64
        self.input_shape : Tuple[int, int, int] = (244, 244, 3)
        self.modelo : Sequential = None
        self.estado : SpoofingEstados = SpoofingEstados.NEW
        self.num_classes = 0
        self.optimizer=optimizer
        self.loss=loss
        self.metrics=metrics
        self.entradas : list = []
        self.labels : list = []
        self.teste_entradas : list = []
        self.teste_labels : list = []
        self.validacao_entradas : list = []
        self.validacao_labels : list = []
        self.X_train : list = []
        self.Y_train : list = []
        self.X_test : list = []
        self.Y_test : list = []
        self.X_valid : list = []
        self.Y_valid : list = []
        self.trainned_epochs = 0
        self.callbacks : List[Callback] = []

        # Redirecionando a Log do TensorFlow para usar nosso sistema de Log logging Python
        tf.get_logger().handlers = logger.handlers
        tf.get_logger().setLevel(logger.level)

        # Exemplo de log do TensorFlow
        tf.get_logger().info("Logger do Tensoflow integrado com sucesso!")
        device_name = tf.test.gpu_device_name()
        if device_name != '/device:GPU:0':
            tf.get_logger().info("NO GPU Found")
        tf.get_logger().info("Found GPU at: %s", device_name)

    def append_callback(self, callback : Callback ):
        self.callbacks.append(callback)


    def load_trainning_images_from_path(self, image_path,
                              label=0, samples=10,
                              tamanho: Tuple[int, int] = (244, 244)):
        entradas = processar_imagem_caminho(image_path, tamanho=tamanho, samples=samples)
        self.entradas.extend( entradas )
        for _ in entradas:
            self.labels.append( label )

    def load_test_images_from_path(self, image_path,
                              label=0, samples=10,
                              tamanho: Tuple[int, int] = (244, 244)):
        teste_entradas = processar_imagem_caminho(image_path, tamanho=tamanho, samples=samples)
        self.teste_entradas.extend( teste_entradas )
        for _ in teste_entradas:
            self.teste_labels.append( label )

    def load_validation_images_from_path(self, image_path,
                              label=0, samples=10,
                              tamanho: Tuple[int, int] = (244, 244)):
        validacao_entradas = processar_imagem_caminho(image_path, tamanho=tamanho, samples=samples)
        self.validacao_entradas.extend( validacao_entradas )
        for _ in validacao_entradas:
            self.validacao_labels.append( label )


    def totalize_images(self, environment="trainning"):
        total_bytes = 0
        count_images = 0
        labels = {}
        entradas = self.entradas
        saidas = self.labels
        if environment != "trainning":
            entradas = self.teste_entradas
            saidas = self.teste_labels
        for index, img in enumerate(entradas):
            total_bytes += img.nbytes
            count_images += 1
            current_label = saidas[index]
            if current_label in labels:
                labels[current_label] += 1
            else:
                labels[current_label] = 1
        return labels, total_bytes

    def loaded_images_summary(self):
        logger.info("*** Loaded images summary ***")

        labels_trainning, total_bytes = self.totalize_images()
        trainning_total_kb = total_bytes / 1024
        trainning_total_mb = trainning_total_kb / 1024
        labels_testing, total_bytes = self.totalize_images(environment="testing")
        testing_total_kb = total_bytes / 1024
        testing_total_mb = testing_total_kb / 1024

        for current_item in labels_trainning.items():
            logger.info(f"Foram carregadas {current_item[1]} imagens " +
                        f"de treinamento com label {current_item[0]}")
        for current_item in labels_testing.items():
            logger.info(f"Foram carregadas {current_item[1]} imagens " +
                        f"de teste com label {current_item[0]}")
        logger.info(f"Memória consumida: Treinamento({trainning_total_mb:.2f}Mb) e " + 
                    f"Teste({testing_total_mb:.2f}Mb)")

    def residual_block(self, filters):
        block = Sequential()
        block.add(Conv2D(filters, (3, 3), padding='same', activation='relu'))
        block.add(Conv2D(filters, (3, 3), padding='same'))
        return block

    def generate(self, input_shape=(244, 244, 3), num_classes=2):
        logger.info("Gerando modelo...")
        self.num_classes = num_classes
        inputs = Input(shape=input_shape)
        x = Conv2D(64, (7, 7), strides=2, padding='same', activation='relu')(inputs)
        x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

        # Residual Block x2 – 64 filtros
        for _ in range(2):
            shortcut = x
            res = self.residual_block(64)(x)
            x = layers.add([res, shortcut])
            x = Activation('relu')(x)

        # Residual Block x2 – 128 filtros
        for _ in range(2):
            shortcut = Conv2D(128, (1, 1), padding='same')(x)
            res = self.residual_block(128)(x)
            x = layers.add([res, shortcut])
            x = Activation('relu')(x)

        # Residual Block x2 – 256 filtros
        for _ in range(2):
            shortcut = Conv2D(256, (1, 1), padding='same')(x)
            res = self.residual_block(256)(x)
            x = layers.add([res, shortcut])
            x = Activation('relu')(x)

        # Residual Block x2 – 512 filtros
        for _ in range(2):
            shortcut = Conv2D(512, (1, 1), padding='same')(x)
            res = self.residual_block(512)(x)
            x = layers.add([res, shortcut])
            x = Activation('relu')(x)

        # DropBlock substituído por Dropout (DropBlock requer lib externa)
        # x = DropBlock(block_size=7, drop_prob=0.2)(x)
        # x = Dropout(0.3)(x)
        x = DropBlock2D(block_size=5, drop_prob=0.3)(x)

        x = GlobalAveragePooling2D()(x)
        outputs = Dense(self.num_classes, activation='sigmoid')(x)

        self.modelo = Model(inputs, outputs)
        self.estado = SpoofingEstados.GENERATED
        return self.modelo

    def compile(self) -> None:
        """Compila o modelo"""
        logger.info("Compilando o modelo...")
        if self.estado >= SpoofingEstados.GENERATED \
        and self.modelo is not None:
            self.modelo.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
            self.modelo.summary()
            entradas = np.array(self.entradas)
            teste_entradas = np.array(self.teste_entradas)
            validacao_entradas = np.array(self.validacao_entradas)
            trainning_labels_size = len(self.entradas)
            test_labels_size = len(self.teste_entradas)
            all_labels = np.concatenate((np.array(self.labels), 
                                         np.array(self.teste_labels), 
                                         np.array(self.validacao_labels)))
            all_labels_categoricos = to_categorical(all_labels, num_classes=self.num_classes)
            train_labels_categoricos = all_labels_categoricos[:trainning_labels_size]
            teste_labels_categoricos = all_labels_categoricos[trainning_labels_size:\
                                                              trainning_labels_size +\
                                                                test_labels_size]
            validacao_labels_categoricos = all_labels_categoricos[trainning_labels_size +\
                                                                  test_labels_size:]
            # self.entradas[np.newaxis, ...]
            # self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            #     entradas, labels_categoricos, random_state=random_state, test_size=test_size)
            self.X_train = entradas
            self.Y_train = train_labels_categoricos
            self.X_test = teste_entradas
            self.Y_test = teste_labels_categoricos
            self.X_valid = validacao_entradas
            self.Y_valid = validacao_labels_categoricos
            self.estado = SpoofingEstados.COMPILED
            logger.info(f"Formato da entrada treinamento: {self.X_train.shape}")
            logger.info(f"Formato da saida treinamento: {self.Y_train.shape}")
            logger.info(f"Formato da entrada teste: {self.X_test.shape}")
            logger.info(f"Formato da saida teste: {self.Y_test.shape}")
            logger.info(f"Formato da entrada validacao: {self.X_valid.shape}")
            logger.info(f"Formato da saida validacao: {self.Y_valid.shape}")

        else:
            raise SpoofingException("Modelo precisa ser gerado primeiro")

    def trainning(self, epochs=10, batch_size=32) -> list:
        """Faz o treinamento do modelo"""
        logger.info("Treinando o modelo...")
        history = []
        self.trainned_epochs = epochs
        if self.estado >= SpoofingEstados.COMPILED \
        and self.modelo is not None:
            history = self.modelo.fit(self.X_train, self.Y_train, 
                                      validation_data=(self.X_valid,  self.Y_valid),
                                      epochs=epochs, batch_size=batch_size, 
                                      callbacks=self.callbacks)
            self.estado = SpoofingEstados.TRAINED
        else:
            raise SpoofingException("Modelo precisa ser compilado primeiro")
        return history

    def evaluating(self) -> list:
        """Faz o teste do modelo"""
        logger.info("Avaliando o modelo com os dados de teste...")
        metrics = []
        if self.estado >= SpoofingEstados.TRAINED \
        and self.modelo is not None:
            metrics = self.modelo.evaluate(self.X_test, self.Y_test)
            self.estado = SpoofingEstados.EVALUATED
        else:
            raise SpoofingException("Modelo precisa ser treinado primeiro")
        return metrics

    def save_state(self, root_weight_file : str) -> None:
        """Salva os pesos do modelo"""
        if self.estado >= SpoofingEstados.TRAINED \
        and self.modelo is not None:
            model_file_name = f"model-samples-{len(self.entradas)}" +\
                f"-epochs-{self.trainned_epochs}.weights.h5"
            full_path_model_file_name = os.path.join(root_weight_file, model_file_name)
            logger.info(f"Gravando o modelo no arquivo {full_path_model_file_name}")
            self.modelo.save_weights(full_path_model_file_name)
        else:
            raise SpoofingException("Modelo precisa ser treinado primeiro")


def load_samples(subsets : List[FaceClassSubSet], samples=10,
                 load_function : Optional[Callable[[str, Union[str, int], int], 
                                                   None]] = None) -> None:
    for subset in subsets:
        for instance_index in range(subset["instance_start"], subset["instance_end"]):
            for version_index in range(subset["version_start"], subset["version_end"]):
                wildcard = f"{subset['relative_path']}{subset['prefix']}{instance_index}"
                wildcard += f"{subset['version_prefix']}{version_index}*.*"
                relative_file_path = os.path.join(SPOOF_CASIA_FASD_PATH, wildcard)
                logger.debug(f"Carregando imagens de: {relative_file_path}")
                load_function(relative_file_path, subset["label"], samples)
    # print(f"Foram carregadas: {len(spoofing.entradas)} imagens")

def load_test_samples(spoofing : Spoofing, test_samples=10, validation_samples=10):
    subsets = [
        {"relative_path": "test\\live\\",
         "prefix": "s", "label": 0, "instance_start": 1, "instance_end": 30,
         "version_start": 1, "version_end": 2, "version_prefix": "v"},
        {"relative_path": "test\\spoof\\",
         "prefix": "s", "label": 1, "instance_start": 1, "instance_end": 30,
         "version_start": 3, "version_end": 8, "version_prefix": "v"},
        {"relative_path": "test\\spoof\\",
         "prefix": "s", "label": 1, "instance_start": 1, "instance_end": 30,
         "version_start": 1, "version_end": 4, "version_prefix": "vHR_"}
    ]
    logger.info("Carregando imagens de teste")
    load_samples(subsets=subsets, samples=test_samples,
                load_function=spoofing.load_test_images_from_path)
    load_samples(subsets=subsets, samples=validation_samples,
                load_function=spoofing.load_validation_images_from_path)    

def load_trainning_samples( spoofing : Spoofing, samples=10 ):
    subsets = [
        {"relative_path": "train\\live\\",
         "prefix": "bs", "label": 0, "instance_start": 1, "instance_end": 20,
         "version_start": 1, "version_end": 2, "version_prefix": "v"},
        {"relative_path": "train\\live\\",
         "prefix": "fs", "label": 0, "instance_start": 1, "instance_end": 20,
         "version_start": 1, "version_end": 2, "version_prefix": "v"},
        {"relative_path": "train\\live\\",
         "prefix": "s", "label": 0, "instance_start": 1, "instance_end": 20,
         "version_start": 1, "version_end": 2, "version_prefix": "v"},
        {"relative_path": "train\\spoof\\",
         "prefix": "s", "label": 1, "instance_start": 1, "instance_end": 20,
         "version_start": 3, "version_end": 8, "version_prefix": "v"},
        {"relative_path": "train\\spoof\\",
         "prefix": "s", "label": 1, "instance_start": 1, "instance_end": 20,
         "version_start": 1, "version_end": 4, "version_prefix": "vHR_"}
    ]
    logger.info("Carregando imagens de treinamento")
    load_samples(subsets=subsets, samples=samples,
                load_function=spoofing.load_trainning_images_from_path)

def main():
    """Função principal"""
    trainning_samples=2
    test_samples=1
    validation_samples=1
    epochs=1
    logger.info("Treinamento do sistema de identificação de Spoofing")
    spoof = Spoofing(optimizer=RMSprop(learning_rate=0.00001, clipvalue=1.0),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
    # input("Tecle <ENTER> para continuar")
    # spoof.load_images_from_path(SPOOF_CASIA_FASD_PATH + "\\train\\live",
    #                             label=0, samples=each_samples)
    # spoof.load_images_from_path(SPOOF_CASIA_FASD_PATH + "\\train\\spoof",
    #                             label=1, samples=each_samples)
    load_trainning_samples(spoofing=spoof, samples=trainning_samples)
    load_test_samples(spoofing=spoof, test_samples=test_samples, 
                      validation_samples=validation_samples)
    spoof.loaded_images_summary()
    # input("Tecle [ENTER] para iniciar o treinamento ou [CTRL] + [C] para cancelar")
    spoof.generate()
    spoof.compile()
    spoof.append_callback(EERHTERCallback(
        validation_data=(spoof.X_valid, spoof.Y_valid), 
        logger=logger)
    )
    trainning_result = spoof.trainning(epochs=epochs, batch_size=32)
    logger.info(f"Trainning Accuracy: {trainning_result.history['accuracy']}")
    logger.info(f"Trainning Loss: {trainning_result.history['loss']}")
    eval_loss, eval_accuracy = spoof.evaluating()
    logger.info(f"Evaluate Accuracy: {eval_accuracy}")
    logger.info(f"Evaluate Loss: {eval_loss}")
    spoof.save_state(SPOOF_MODEL_PATH)
    logger.info("Modelo Gerado e Salvo")
