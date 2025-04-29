# spoofing/metricas_adicionais.py
""""Módulo usado para declarar o Callback que mede as metricas EER e HTER"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve
from tensorflow.keras.callbacks import Callback

class EERHTERCallback(Callback):
    def __init__(self, validation_data, logger=None):
        super().__init__()
        self.validation_data = validation_data  # (X_val, y_val)
        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data

        # Faz as predições
        y_pred = self.model.predict(X_val)
        
        if y_pred.shape[-1] > 1:
            # Se for softmax, pegar probabilidade da classe 'real'
            y_pred = y_pred[:, 1]

        # ROC Curve para obter FAR e FRR
        if len(y_val.shape) > 1 and y_val.shape[1] > 1:
            y_val = np.argmax(y_val, axis=1)
        fpr, tpr, thresholds = roc_curve(y_val, y_pred)

        fnr = 1 - tpr  # False Negative Rate

        # EER é onde FPR ~= FNR
        eer_threshold_idx = np.nanargmin(np.absolute((fnr - fpr)))
        eer = (fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2.0

        # HTER é (FAR + FRR)/2 usando o mesmo threshold
        hter = (fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2.0

        message = f"Epoch {epoch+1}: EER = {eer:.4f}, HTER = {hter:.4f}"

        if self.logger:
            self.logger.info(message)
        else:
            print(message)

        # Opcional: pode adicionar no logs para TensorBoard, etc.
        logs['eer'] = eer
        logs['hter'] = hter
