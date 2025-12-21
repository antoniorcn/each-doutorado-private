import tensorflow as tf
import numpy as np
import pygame
import threading
import time

# === Configurações da imagem ===
IMG_SIZE = 244

# === Gerar uma imagem de exemplo e rótulo ===
input_image = np.random.rand(IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
input_label = np.array([1.0])  # Classe positiva

# Adiciona batch dimension
input_batch = np.expand_dims(input_image, axis=0)

# === Variável para armazenar saída da convolução ===
convoluted_output = None
lock = threading.Lock()

# === Criar o modelo com Input explícito ===
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu')(inputs)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy')

# === Callback para capturar saída da convolução ===
class ConvolutionVisualizer(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoc, logs=None):
        global convoluted_output
        # Cria modelo intermediário com saída da primeira camada
        intermediate_model = tf.keras.Model(inputs=model.input,
                                            outputs=model.layers[1].output)  # camada Conv2D
        conv_out = intermediate_model.predict(input_batch)
        with lock:
            convoluted_output = conv_out[0, :, :, 0]  # primeiro canal
    

# === Função de treinamento ===
def train_model():
    model.fit(input_batch, input_label,
              epochs=150,
              batch_size=1,
              callbacks=[ConvolutionVisualizer()])
    time.sleep(0.4)

# === Inicializar Pygame ===
pygame.init()
screen = pygame.display.set_mode((IMG_SIZE * 2, IMG_SIZE))
pygame.display.set_caption("Visualização da Convolução")


def update_pygame():
    font_render = pygame.font.SysFont("arial", 32, False, False)
    frame_id = 0
    running = True
    while running:
        screen.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                return

        # Desenha imagem original
        original_surface = pygame.surfarray.make_surface((input_image * 255).astype(np.uint8))
        screen.blit(original_surface, (0, 0))

        # Desenha imagem convolucionada
        with lock:
            if convoluted_output is not None:
                conv_img = convoluted_output
                conv_img = (conv_img - np.min(conv_img)) / (np.max(conv_img) - np.min(conv_img) + 1e-5)
                conv_rgb = np.stack([conv_img]*3, axis=-1)
                conv_rgb = (conv_rgb * 255).astype(np.uint8)
                conv_surface = pygame.surfarray.make_surface(conv_rgb)
                conv_surface = pygame.transform.scale(conv_surface, (IMG_SIZE, IMG_SIZE))
                screen.blit(conv_surface, (IMG_SIZE, 0))
            img_frame = font_render.render(f"Frame: {frame_id}", False, (255, 255, 0))
            screen.blit(img_frame, (screen.get_width() - 200, 10))
            frame_id += 1
            
        pygame.display.update()
        time.sleep(0.05)

# === Rodar tudo ===
train_thread = threading.Thread(target=train_model)
pygame_thread = threading.Thread(target=update_pygame)

train_thread.start()
pygame_thread.start()

train_thread.join()
pygame_thread.join()
