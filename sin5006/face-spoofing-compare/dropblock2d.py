import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable(package="Custom")
class DropBlock2D(Layer):
    def __init__(self, block_size=7, drop_prob=0.3, **kwargs):
        super(DropBlock2D, self).__init__(**kwargs)
        self.block_size = block_size
        self.drop_prob = drop_prob
        self.input_shape_  = None

    def build(self, input_shape):
        self.input_shape_ = input_shape
        super(DropBlock2D, self).build(input_shape)

    def compute_gamma(self, feat_shape):
        # Tamanho da feature map
        # h, w = feat_shape[1], feat_shape[2]
        # Evita divisão por zero
        # keep_prob = 1.0 - self.drop_prob
        block_area = self.block_size ** 2
        # valid_area = (h - self.block_size + 1) * (w - self.block_size + 1)
        gamma = self.drop_prob * tf.cast(tf.size(feat_shape[1:]), tf.float32) / tf.cast(block_area, tf.float32)
        gamma = tf.clip_by_value(gamma, 0.0, 1.0)
        return gamma

    def call(self, inputs, training=None):
        if not training or self.drop_prob == 0.0:
            return inputs

        # Shape: (batch, height, width, channels)
        input_shape = tf.shape(inputs)
        batch_size, height, width, channels = input_shape[0], input_shape[1], input_shape[2], input_shape[3]

        # Calcula gamma de maneira segura
        gamma = self.compute_gamma(inputs.shape)

        # Gera máscara booleana com Bernoulli(gamma)
        mask = tf.cast(tf.random.uniform((batch_size, height, width, channels)) < gamma, tf.float32)

        # Expande para blocos
        def compute_block_mask(mask):
            # Cria kernel com blocos de 1s
            kernel = tf.ones((self.block_size, self.block_size, channels, 1), dtype=tf.float32)
            # Aplica convolução para expandir os blocos
            mask = tf.nn.depthwise_conv2d(
                mask,
                filter=kernel,
                strides=[1, 1, 1, 1],
                padding='SAME'
            )
            mask = tf.clip_by_value(mask, 0.0, 1.0)  # Binário
            return 1.0 - mask

        block_mask = compute_block_mask(mask)

        # Reescala para manter o valor esperado
        keep_ratio = tf.reduce_sum(block_mask) / tf.cast(tf.size(block_mask), tf.float32)
        block_mask = block_mask / (keep_ratio + 1e-6)

        return inputs * block_mask

    def get_config(self):
        config = super(DropBlock2D, self).get_config()
        config.update({
            'block_size': self.block_size,
            'drop_prob': self.drop_prob
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
