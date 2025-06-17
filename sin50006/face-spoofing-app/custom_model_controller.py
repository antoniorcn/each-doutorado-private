import numpy as np
from tensorflow.keras import models

class CustomModelController:
    def __init__(self, **kwargs):
        self.load_model_to_memory()

    def load_model_to_memory(self):
        self.model = models.load_model('./model-samples-36828-epochs-50.weights.h5')

        # Verificar o sumário do modelo
        self.model.summary()

    def is_real_face(self, face_batch):
        result = self.model.predict(face_batch)
        print("Resultado: ", result)
        return result[0][0] > 0.5