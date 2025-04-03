import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from .processador_image import processar_imagem
from enum import Enum

# Definindo uma Enumeração
class SpoofingEstados(Enum):
    NEW = 1
    GENERATED = 2
    COMPILED = 3
    TRAINED = 4
    READY = 5

class SpoofingException(Exception):
    """Classe de Excepção para Spoofing"""
    def __init__(self, mensagem):
        super().__init__(mensagem)
        self.mensagem = mensagem


class Spoofing:
    """Classe Spoofing que gera, compila, treina e salva os pesos gerados"""
    def __init__(self, image_path : str):
        self.kernel_size = (7, 7)
        self.strides = (2, 2)
        self.filters = 64
        self.input_shape= (244, 244, 3)
        self.modelo : Sequential = None
        self.estado : SpoofingEstados = SpoofingEstados.NEW
        self.image_path = image_path

    def generate(self) -> Sequential:
        """Gera o modelo com as camadas definidas no projeto"""
        seq = Sequential()
        seq.add(Conv2D( filters=self.filters,
                        kernel_size=self.kernel_size,
                        strides=self.strides,
                        activation='relu', input_shape=self.input_shape))
        # Apenas para teste
        seq.add(Conv2D(64, (3, 3), activation='relu'))
        seq.add(Flatten())
        seq.add(Dense(128, activation='relu'))
        seq.add(Dense(10, activation='softmax'))

        self.modelo = seq
        self.estado = SpoofingEstados.GENERATED
        return seq

    def compile(self) -> None:
        """Compila o modelo"""
        if self.estado >= SpoofingEstados.GENERATED \
        and self.modelo is not None: 
            self.modelo.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
            self.modelo.summary()
            self.estado = SpoofingEstados.COMPILED
        else:
            raise SpoofingException("Modelo precisa ser gerado primeiro")

    def trainning(self) -> None:
        """Faz o treinamento do modelo"""
        if self.estado >= SpoofingEstados.COMPILED \
        and self.modelo is not None: 
            entrada = processar_imagem(self.image_path)
            # Adiciona uma dimensão para representar o batch
            entrada = entrada[np.newaxis, ...]  
            
            label = to_categorical([2], num_classes=10)  # Classe 2 em one-hot

            self.modelo.fit(entrada, label, epochs=10)
            self.estado = SpoofingEstados.TRAINED
        else:
            raise SpoofingException("Modelo precisa ser compilado primeiro")

    def save_state(self, result_weight_file : str) -> None:
        """Salva os pesos do modelo"""
        if self.estado >= SpoofingEstados.TRAINED \
        and self.modelo is not None:
            self.modelo.save_weights(result_weight_file)
        else:
            raise SpoofingException("Modelo precisa ser treinado primeiro")


def main():
    """Função principal"""
    print("Hello World")
    spoof = Spoofing(image_path="")
    spoof.generate()
    spoof.compile()
    spoof.trainning()
    spoof.save_state("pesos_modelo.h5")
    print("Modelo Gerado")