import tensorflow as tf

class CustomConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', use_bias=True, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding = padding.upper()
        self.use_bias = use_bias
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        in_channels = input_shape[-1]

        # Kernel: [kernel_height, kernel_width, in_channels, out_channels]
        self.kernel = self.add_weight(
            shape=(self.kernel_size[0], self.kernel_size[1], in_channels, self.filters),
            initializer='glorot_uniform',
            trainable=True,
            name='kernel'
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.filters,),
                initializer='zeros',
                trainable=True,
                name='bias'
            )
        else:
            self.bias = None

    def call(self, inputs):
        x = tf.nn.conv2d(
            inputs,
            filters=self.kernel,
            strides=(1, *self.strides, 1),
            padding=self.padding
        )
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)

        if self.activation is not None:
            x = self.activation(x)
        return x
