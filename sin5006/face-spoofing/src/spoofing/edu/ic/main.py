import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import numpy as np
import random
import tensorflow as tf

from jproperties import Properties
from typing import List, Optional, TypedDict, Callable, Tuple, Union
from enum import IntEnum
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D,\
    Dense, Dropout, Add, Activation
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import Callback
from spoofing.edu.ic.processador_image import criar_dataset, calcular_tamanho_bytes, dataset_length, carregar_imagens_com_label
from spoofing.edu.ic.dropblock2d import DropBlock2D
from spoofing.edu.ic.logger import get_logger_arquivo
from spoofing.edu.ic.metricas_adicionais import EERHTERCallback, F1Score

logger = get_logger_arquivo(__name__)
logger.info("Detector inicializado")
logger.info(f"Diretório atual: {os.getcwd()}")
config_path = "./each-doutorado-private/sin50006/face-spoofing/face-spoofing.properties"
if not os.path.exists(config_path):
    logger.error(f"Arquivo '{config_path}' não encontrado.")
    exit(0)

config = Properties()
with open(config_path, 'rb') as config_file:
    config.load(config_file)

logger.info(f"Lendo informacoes de {config_path}")

# Iterate through sections and properties
for item_view in config.items():
    logger.info(f"{item_view[0]}={item_view[1].data}")

optimizer_name = config.get("optimizer_name").data
loss_function = config.get("loss_function").data
epochs=int(config.get("epochs").data)
batch_size=int(config.get("batch_size").data)
drop_block_size=int(config.get("drop_block_size").data)
drop_block_prob=float(config.get("drop_block_prob").data)

trainning_percentage_samples=float(config.get("trainning_percentage_samples").data)
test_percentage_samples=float(config.get("test_percentage_samples").data)
validation_percentage_samples=float(config.get("validation_percentage_samples").data)

def get_env_var(name: str, default: str = None) -> str:
    """Lê uma variável de ambiente com nome `name`.
       Se não existir, retorna o `default`.
    """
    value = os.getenv(name, default)
    return value

SPOOF_DADOS_PATH = config.get("data_path").data
SPOOF_CASIA_FASD_PATH = config.get("casia_fasd_path").data
SPOOF_MODEL_PATH = config.get("model_path").data
# SPOOF_MODEL_PATH = os.path.join(SPOOF_DADOS_PATH, "face-spoofing/modelos")

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
    def __init__(self, optimizer="adam", loss='binary_crossentropy', local_batch_size=batch_size):
        self.kernel_size : Tuple[int, int] = (7, 7)
        self.strides : Tuple[int, int] = (2, 2)
        self.filters : int = 64
        self.input_shape : Tuple[int, int, int] = (244, 244, 3)
        self.modelo : Sequential = None
        self.estado : SpoofingEstados = SpoofingEstados.NEW
        self.num_classes = 0
        self.optimizer=optimizer
        self.loss=loss
        self.treinamento : tf.data.Dataset[Tuple[tf.Tensor, tf.Tensor]] = None
        self.validacao : tf.data.Dataset[Tuple[tf.Tensor, tf.Tensor]] = None
        self.teste : tf.data.Dataset[Tuple[tf.Tensor, tf.Tensor]] = None
        self.trainning_class_weight_dict = dict()
        self.batch_size = local_batch_size
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


    def load_trainning_images_from_path(self, image_paths_labels : List[Tuple[str, int]],
                              tamanho: Tuple[int, int] = (244, 244)):
        labels = [label for _, label in image_paths_labels]
        labels_classes = np.unique(labels)
        class_weights = compute_class_weight('balanced', classes=labels_classes, y=labels)
        self.trainning_class_weight_dict = dict(enumerate(class_weights))
        logger.info(f"Pesos gerados para as classes de treinamento: {self.trainning_class_weight_dict}")
        dataset = criar_dataset(image_paths_labels, tamanho=tamanho, batch_size=self.batch_size)
        logger.info(f"Images no dataset de treinamento criado: {sum(dataset_length(dataset)[1])}")
        if self.treinamento is None:
            self.treinamento = dataset
        else:
            self.treinamento = self.treinamento.concatenate( dataset )
        logger.info(f"Images totais no dataset de treinamento: {sum(dataset_length(self.treinamento)[1])}")
        return dataset_length(self.treinamento)

    def load_test_images_from_path(self, image_paths_labels : List[Tuple[str, int]],
                              tamanho: Tuple[int, int] = (244, 244)):
        dataset = criar_dataset(image_paths_labels, tamanho=tamanho, batch_size=self.batch_size)
        logger.info(f"Images no dataset de testes criado: {sum(dataset_length(dataset)[1])}")
        if self.teste is None:
            self.teste = dataset
        else:
            self.teste = self.teste.concatenate( dataset )
        logger.info(f"Images totais no dataset de testes: {sum(dataset_length(self.teste)[1])}")
        return dataset_length(self.teste)

    def load_validation_images_from_path(self, image_paths_labels : List[Tuple[str, int]],
                              tamanho: Tuple[int, int] = (244, 244)):
        dataset = criar_dataset(image_paths_labels, tamanho=tamanho, batch_size=self.batch_size)
        logger.info(f"Images no dataset de validacao criado: {sum(dataset_length(dataset)[1])}")
        if self.validacao is None:
            self.validacao = dataset
        else:
            self.validacao = self.validacao.concatenate( dataset )
        logger.info(f"Images totais no dataset de validacao: {sum(dataset_length(self.validacao)[1])}")            
        return dataset_length(self.validacao)


    def totalize_images(self, environment="trainning"):
        total_bytes = 0
        count_images = 0
        labels = {}
        entradas = self.treinamento
        if environment == "testing":
            entradas = self.teste
        elif environment == "validation":
            entradas = self.validacao
        for (imagem, label) in entradas.unbatch():
            total_bytes += calcular_tamanho_bytes(imagem)
            count_images += 1
            label_val = int(label.numpy())
            if label_val in labels:
                labels[label_val] += 1
            else:
                labels[label_val] = 1
        return labels, total_bytes

    def loaded_images_summary(self):
        logger.info("*** Loaded images summary ***")

        labels_trainning, total_bytes = self.totalize_images()
        trainning_total_kb = total_bytes / 1024
        trainning_total_mb = trainning_total_kb / 1024
        labels_testing, total_bytes = self.totalize_images(environment="testing")
        testing_total_kb = total_bytes / 1024
        testing_total_mb = testing_total_kb / 1024
        labels_validation, total_bytes = self.totalize_images(environment="validation")
        validation_total_kb = total_bytes / 1024
        validation_total_mb = validation_total_kb / 1024

        logger.info(f"Total imagens de treinamento: {sum(dataset_length(self.treinamento)[1])}")
        for current_item in labels_trainning.items():
            logger.info(f"Foram carregadas {current_item[1]} imagens " +
                        f"de treinamento com label {current_item[0]}")
            
        logger.info(f"Total imagens de testes: {sum(dataset_length(self.teste)[1])}")
        for current_item in labels_testing.items():
            logger.info(f"Foram carregadas {current_item[1]} imagens " +
                        f"de teste com label {current_item[0]}")

        logger.info(f"Total imagens de validacao: {sum(dataset_length(self.validacao)[1])}")
        for current_item in labels_validation.items():
            logger.info(f"Foram carregadas {current_item[1]} imagens " +
                        f"de validacao com label {current_item[0]}")
        logger.info(f"Memória consumida: Treinamento({trainning_total_mb:.2f}Mb), " +
                    f"Teste({testing_total_mb:.2f}Mb) e Validacao({validation_total_mb:.2f}Mb)")

    def residual_block(self, filters):
        block = Sequential()
        block.add(Conv2D(filters, (3, 3), padding='same', activation='relu'))
        block.add(Conv2D(filters, (3, 3), padding='same'))
        return block

    def generate(self, input_shape=(244, 244, 3), num_classes=1):
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

        x = DropBlock2D(block_size=drop_block_size, drop_prob=drop_block_prob)(x)

        x = GlobalAveragePooling2D()(x)
        outputs = Dense(self.num_classes, activation='sigmoid')(x)

        self.modelo = Model(inputs, outputs)
        self.estado = SpoofingEstados.GENERATED
        return self.modelo

    def compile(self,
            metrics : List = [
            AUC(name='auc'),
            Precision(name='precision'),
            Recall(name='recall'),
            F1Score(name='f1_score'),
            "accuracy"
        ]) -> None:
        """Compila o modelo"""
        logger.info("Compilando o modelo...")
        
        if self.estado >= SpoofingEstados.GENERATED \
        and self.modelo is not None:
            self.modelo.compile(optimizer=self.optimizer,
            loss=BinaryCrossentropy(),
            metrics=metrics)
            self.modelo.summary()
            self.estado = SpoofingEstados.COMPILED
            logger.info(f"Formato da entrada e saida do treinamento: {self.treinamento.element_spec}")
            logger.info(f"Formato da entrada e saida do teste: {self.teste.element_spec}")
            logger.info(f"Formato da entrada e saida davalidacao: {self.validacao.element_spec}")

        else:
            raise SpoofingException("Modelo precisa ser gerado primeiro")

    def trainning(self, epochs=10) -> list:
        """Faz o treinamento do modelo"""
        logger.info("Treinando o modelo...")
        history = []
        self.trainned_epochs = epochs
        if self.estado >= SpoofingEstados.COMPILED \
        and self.modelo is not None:
            history = self.modelo.fit(self.treinamento,
                                      validation_data=self.validacao,
                                      epochs=epochs,
                                      callbacks=self.callbacks,
                                      class_weight=self.trainning_class_weight_dict)
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
            metrics = self.modelo.evaluate(self.teste, return_dict=True)
            self.estado = SpoofingEstados.EVALUATED
        else:
            raise SpoofingException("Modelo precisa ser treinado primeiro")
        return metrics

    def save_state(self, root_weight_file : str) -> None:
        """Salva os pesos do modelo"""
        if self.estado >= SpoofingEstados.TRAINED \
        and self.modelo is not None:
            model_file_name = f"model-samples-{sum(dataset_length(self.treinamento)[1])}" +\
                f"-epochs-{self.trainned_epochs}.weights.h5"
            full_path_model_file_name = os.path.join(root_weight_file, model_file_name)
            logger.info(f"Gravando o modelo no arquivo {full_path_model_file_name}")
            self.modelo.save_weights(full_path_model_file_name)
        else:
            raise SpoofingException("Modelo precisa ser treinado primeiro")

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

def load_trainning_samples(trainning_percentage_samples : float=0.10 ):
    subsets = [
         {"relative_path": "casia-fasd-train/live/",
         "prefix": "bs", "label": 0, "instance_start": 1, "instance_end": 20,
         "version_start": 1, "version_end": 2, "version_prefix": "v"},
        {"relative_path": "casia-fasd-train/live/",
         "prefix": "fs", "label": 0, "instance_start": 1, "instance_end": 20,
         "version_start": 1, "version_end": 2, "version_prefix": "v"},
        {"relative_path": "casia-fasd-train/live/",
         "prefix": "s", "label": 0, "instance_start": 1, "instance_end": 20,
         "version_start": 1, "version_end": 2, "version_prefix": "v"},
        {"relative_path": "casia-fasd-train/spoof/",
         "prefix": "s", "label": 1, "instance_start": 1, "instance_end": 20,
         "version_start": 3, "version_end": 8, "version_prefix": "v"},
        {"relative_path": "casia-fasd-train/spoof/",
         "prefix": "s", "label": 1, "instance_start": 1, "instance_end": 20,
         "version_start": 1, "version_end": 4, "version_prefix": "vHR_"}
    ]
    logger.info("Carregando imagens de treinamento")
    trainning_files = load_files_names(subsets)
    logger.info(f"Foram identificadas: {len(trainning_files)} imagens de treinamento")

    quantidade = int(len(trainning_files) * trainning_percentage_samples) + 1
    sample_trainning_files = random.sample(trainning_files, quantidade)
    logger.info(f"Foram carregadas: {len(sample_trainning_files)} imagens de treinamento")


    return sample_trainning_files


def load_test_validation_samples(test_percentage_samples=0.10, validation_percentage_samples=0.10):
    subsets = [
        {"relative_path": "casia-fasd/live/",
         "prefix": "s", "label": 0, "instance_start": 1, "instance_end": 30,
         "version_start": 1, "version_end": 2, "version_prefix": "v"},
        {"relative_path": "casia-fasd/spoof/",
         "prefix": "s", "label": 1, "instance_start": 1, "instance_end": 30,
         "version_start": 3, "version_end": 8, "version_prefix": "v"},
        {"relative_path": "casia-fasd/spoof/",
         "prefix": "s", "label": 1, "instance_start": 1, "instance_end": 30,
         "version_start": 1, "version_end": 4, "version_prefix": "vHR_"}
    ]
    logger.info("Carregando imagens de testes e validacao")
    tests_validation_files = load_files_names(subsets)
    logger.info(f"Foram identificadas: {len(tests_validation_files)} imagens de testes e treinamento")

    quantidade_testes = int(len(tests_validation_files) * test_percentage_samples) + 1
    quantidade_validacao = int(len(tests_validation_files) * validation_percentage_samples) + 1
    quantidade_imagens = quantidade_testes + quantidade_validacao
    sample_test_validation_files = random.sample(tests_validation_files, quantidade_imagens)
    sample_test_files = sample_test_validation_files[:quantidade_testes]
    sample_validation_files = sample_test_validation_files[quantidade_testes:]
    logger.info(f"Foram carregadas: {len(sample_test_files)} imagens de testes")
    logger.info(f"Foram carregadas: {len(sample_validation_files)} imagens de validação")
    return sample_test_files, sample_validation_files

def main():
    """Função principal"""
    logger.info("Treinamento do sistema de identificação de Spoofing")
    rmsprop = RMSprop(learning_rate=0.001, clipvalue=1.0)
    adam = Adam(learning_rate=0.1)
    optimizer = rmsprop
    if optimizer_name == "adam":
        optimizer = "adam"
    spoof = Spoofing(optimizer=optimizer, loss=loss_function)

    trainning_samples = load_trainning_samples(
        trainning_percentage_samples=trainning_percentage_samples)
    test_samples, validation_samples = load_test_validation_samples(
        test_percentage_samples=test_percentage_samples,
        validation_percentage_samples=validation_percentage_samples)

    spoof.load_trainning_images_from_path(trainning_samples, tamanho=(244, 244))
    spoof.load_test_images_from_path(test_samples, tamanho=(244, 244))
    spoof.load_validation_images_from_path(validation_samples, tamanho=(244, 244))

    spoof.loaded_images_summary()

    spoof.generate()
    spoof.compile()
    spoof.append_callback(EERHTERCallback(
        validation_data=spoof.validacao,
        logger=logger)
    )
    trainning_result = spoof.trainning(epochs=epochs)
    logger.info(f"Trainning Accuracy: {trainning_result.history['accuracy']}")
    logger.info(f"Trainning Loss: {trainning_result.history['loss']}")
    logger.info(f"Entire history: {trainning_result.history}")

    evaluation = spoof.evaluating()

    logger.info(f"Evaluations: {evaluation}")
    spoof.save_state(SPOOF_MODEL_PATH)
    logger.info("Modelo Gerado e Salvo")

if __name__ == "__main__":
    main()
