import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from dropblock2d import DropBlock2D
from metricas_adicionais import F1Score, EERHTERCallback

class CustomModelController:
    def __init__(self, **kwargs):
        self.load_model_to_memory()

    def load_model_to_memory(self):
        self.model = models.load_model('./model-samples-7754-epochs-20.weights.keras',
                                       custom_objects={
                                           'DropBlock2D': DropBlock2D,
                                           'F1Score': F1Score,
                                           'EERHTERCallback': EERHTERCallback
                                           })

        # Verificar o sumário do modelo
        self.model.summary()

    def is_real_face(self, face_batch):
        result = self.model.predict(face_batch)
        print("Resultado do modelo customizado: ", result)
        return result[0][0] <= 0.5
        # 0-Live        1-Spoofing