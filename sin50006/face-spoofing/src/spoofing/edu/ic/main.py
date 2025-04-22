from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Add, Activation
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, Model
from spoofing.edu.ic.processador_image import processar_imagem_caminho
from spoofing.edu.ic.dropblock2d import DropBlock2D
from spoofing.edu.ic.logger import get_logger
from enum import IntEnum

logger = get_logger(__name__)
logger.info("Detector inicializado")

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
        self.X_train : list = []
        self.Y_train : list = []
        self.X_test : list = []
        self.Y_test : list = []
        self.trainned_epochs = 0

    def load_images_from_path(self, image_path,
                              tamanho: Tuple[int, int] = (244, 244),
                              label=0, samples=10):
        entradas = processar_imagem_caminho(image_path, tamanho=tamanho, samples=samples)
        self.entradas.extend( entradas )
        for _ in entradas:
            self.labels.append( label )

    def residual_block(self, filters):
        block = Sequential()
        block.add(Conv2D(filters, (3, 3), padding='same', activation='relu'))
        block.add(Conv2D(filters, (3, 3), padding='same'))
        return block

    def generate(self, input_shape=(244, 244, 3), num_classes=2):
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

    def compile(self, random_state, test_size=0.2) -> None:
        """Compila o modelo"""
        if self.estado >= SpoofingEstados.GENERATED \
        and self.modelo is not None:
            self.modelo.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
            self.modelo.summary()
            entradas = np.array(self.entradas)
            labels_categoricos = to_categorical(np.array(self.labels), num_classes=self.num_classes)
            # self.entradas[np.newaxis, ...]
            logger.debug(f"Formato da entrada: {entradas.shape}")
            logger.debug(f"Formato da saida: {labels_categoricos.shape}")
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
                entradas, labels_categoricos, random_state=random_state, test_size=test_size)
            self.estado = SpoofingEstados.COMPILED
        else:
            raise SpoofingException("Modelo precisa ser gerado primeiro")

    def trainning(self, epochs=10) -> list:
        """Faz o treinamento do modelo"""
        logger.debug("### Treinando a rede")
        history = []
        self.trainned_epochs = epochs
        if self.estado >= SpoofingEstados.COMPILED \
        and self.modelo is not None:
            history = self.modelo.fit(self.X_train, self.Y_train, epochs=epochs)
            self.estado = SpoofingEstados.TRAINED
        else:
            raise SpoofingException("Modelo precisa ser compilado primeiro")
        return history

    def evaluating(self) -> list:
        """Faz o teste do modelo"""
        logger.debug("### Avaliando os dados de treinamento a rede")
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
            file_name = f"{root_weight_file}-samples-{len(self.entradas)}" +\
                f"-epochs-{self.trainned_epochs}.weights.h5"
            self.modelo.save_weights(file_name)
        else:
            raise SpoofingException("Modelo precisa ser treinado primeiro")


def main():
    """Função principal"""
    each_samples=10
    epochs=10
    logger.info("Treinamento do sistema de identificação de Spoofing")
    spoof = Spoofing(optimizer=Adam(learning_rate=0.00001, clipvalue=1.0),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
    spoof.load_images_from_path("C:\\git\\dados\\sin50006\\casia-fasd\\train\\live",
                                label=0, samples=each_samples)
    spoof.load_images_from_path("C:\\git\\dados\\sin50006\\casia-fasd\\train\\spoof",
                                label=1, samples=each_samples)
    spoof.generate()
    spoof.compile(random_state=100, test_size=0.2)
    history = spoof.trainning(epochs=epochs)
    logger.info(f"History: {history}")
    metrics = spoof.evaluating()
    logger.info(f"Metrics: {metrics}")
    spoof.save_state("C:\\git\\dados\\sin50006\\pesos_modelo")
    logger.info("Modelo Gerado e Salvo")
