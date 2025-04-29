import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D
from tensorflow.keras import Sequential
import numpy as np
from spoofing.edu.ic.logger import get_logger_arquivo

logger = get_logger_arquivo(__name__)

class DropBlock(Layer):
    def __init__(self, block_size, drop_prob):
        super(DropBlock, self).__init__()
        self.block_size = tf.cast(block_size, tf.float32)
        self.drop_prob = tf.cast(drop_prob, tf.float32)

    def call(self, inputs, training=False):
        if not training:
            return inputs

        input_shape = tf.shape(inputs)
        logger.debug(f"Tipo do input_shape: {input_shape.dtype}")
        height_int, width_int = tf.cast(input_shape[1], tf.int32), tf.cast(input_shape[2], tf.int32)
        height, width = tf.cast(height_int, tf.float32), tf.cast(width_int, tf.float32)
        
        batch_size, channels = input_shape[0], input_shape[3]

        # Calculate the mask for DropBlock
        gamma = tf.cast(self.drop_prob * (height * width) / (self.block_size ** 2), tf.float32)

        mask = tf.cast(tf.random.uniform(shape=[batch_size, height_int, width_int, channels], dtype=tf.float32) < gamma, tf.int32)
        logger.debug(f"Mask: {mask}")
        mask_float = tf.cast(mask, tf.float32)
        logger.debug(f"Mask Float: {mask_float}")
        # Apply block mask
        block_mask = tf.image.resize(mask_float, (height // self.block_size, width // self.block_size))
        block_mask = tf.image.resize(block_mask, (height, width))
        
        # Rescale the mask so that the number of active units is kept constant
        block_mask = 1 - block_mask
        block_mask = block_mask * (height * width) / tf.reduce_sum(block_mask)

        return inputs * block_mask

