import numpy as np
from tensorflow.keras import models
from dropblock2d import DropBlock2D
from metricas_adicionais import F1Score, EERHTERCallback

class CustomModelController:
    """
        Controlador para uso do modelo treinado para identificação de Spoofing.
        Resultado 1 indica Spoofing e 0 indica imagem real
    """
    def __init__(self, **kwargs):
        self.load_model_to_memory()

    def load_model_to_memory(self):
        self.model = models.load_model(
            './model-samples-7754-epochs-20.weights.keras',
            custom_objects={
                'DropBlock2D': DropBlock2D,
                'F1Score': F1Score
                })

        # Verificar o sumário do modelo
        self.model.summary()

    def is_real_face(self, face_image):
        input_data = face_image.astype('float32') / 255.0
        input_data = np.expand_dims(input_data, axis=0)  # (1, 244, 244, 3)
        result = self.model.predict(input_data)
        print("Resultado: ", result)
        return result[0][0] <= 0.5
        # Resultado 1 - Spoofing e 0 - Real
