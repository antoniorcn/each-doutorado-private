# spoofing/metricas_adicionais.py
""""Módulo usado para declarar o Callback que mede as metricas EER e HTER"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve
from tensorflow.keras.callbacks import Callback


class EERHTERCallback(Callback):
    def __init__(self, validation_data, logger=None):
        super().__init__()
        self.validation_data = validation_data
        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        y_true = []
        y_scores = []

        for x_batch, y_batch in self.validation_data:
            y_pred = self.model(x_batch, training=False)
            if y_pred.shape[-1] > 1:
                y_pred = y_pred[:, 1]  # prob da classe real (classe 1)

            y_scores.extend(y_pred.numpy().flatten())
            y_true.extend(y_batch.numpy().flatten())

        y_true = np.array(y_true)
        y_scores = np.array(y_scores)

        # Se for one-hot
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)

        # ROC
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        fnr = 1 - tpr

        if np.all(np.isnan(fpr)) or np.all(np.isnan(fnr)):
            eer = hter = eer_threshold = float('nan')
        else:
            eer_idx = np.nanargmin(np.abs(fpr - fnr))
            eer_threshold = thresholds[eer_idx]
            eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0

            # Classificação binária usando o threshold
            y_pred_class = (y_scores >= eer_threshold).astype(int)
            fp = np.sum((y_pred_class == 1) & (y_true == 0))
            fn = np.sum((y_pred_class == 0) & (y_true == 1))
            tp = np.sum((y_pred_class == 1) & (y_true == 1))
            tn = np.sum((y_pred_class == 0) & (y_true == 0))

            far = fp / (fp + tn + 1e-8)
            frr = fn / (fn + tp + 1e-8)
            hter = (far + frr) / 2.0

        message = f"Epoch {epoch+1}: EER = {eer:.4f}, HTER = {hter:.4f}, Threshold = {eer_threshold:.4f}"

        if self.logger:
            self.logger.info(message)
        else:
            print(message)

        if logs is not None:
            logs['eer'] = eer
            logs['hter'] = hter
            logs['eer_threshold'] = eer_threshold