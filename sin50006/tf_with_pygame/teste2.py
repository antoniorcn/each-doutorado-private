import numpy as np
import pygame
import threading
import time

# === Parâmetros ===
IMG_SIZE = 244
KERNEL_SIZE = 5
LEARNING_RATE = 0.001

# === Gerar imagem de entrada (RGB) e imagem alvo (escala de cinza aleatória) ===
input_image = np.random.rand(IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
target_output = np.random.rand(IMG_SIZE - KERNEL_SIZE + 1, IMG_SIZE - KERNEL_SIZE + 1).astype(np.float32)

# === Inicializar kernel (filtro 5x5x3) ===
kernel = np.random.randn(KERNEL_SIZE, KERNEL_SIZE, 3).astype(np.float32)

# === Função de convolução manual ===
def manual_convolution2d(image, kernel):
    h, w, c = image.shape
    kh, kw, kc = kernel.shape
    assert c == kc, "Canais incompatíveis"
    output = np.zeros((h - kh + 1, w - kw + 1), dtype=np.float32)

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            region = image[i:i+kh, j:j+kw, :]
            output[i, j] = np.sum(region * kernel)

    return output

# === Função para normalizar e converter em imagem Pygame ===
def to_pygame_surface(img, size=None):
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    img = (img * 255).astype(np.uint8)
    surf = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
    if size:
        surf = pygame.transform.scale(surf, size)
    return surf

# === Treinamento manual ===
def train_loop():
    global kernel
    for epoch in range(200):
        output = manual_convolution2d(input_image, kernel)
        error = target_output - output
        loss = np.mean(error ** 2)

        # Atualizar kernel (pseudo-gradiente)
        for i in range(KERNEL_SIZE):
            for j in range(KERNEL_SIZE):
                for c in range(3):
                    patch = input_image[i:i+error.shape[0], j:j+error.shape[1], c]
                    grad = -2 * np.mean(patch * error)
                    kernel[i, j, c] -= LEARNING_RATE * grad

                    # Atualiza visualização
                    with lock:
                        shared_data["conv_output"] = output.copy()
                        shared_data["kernel"] = kernel.copy()

        time.sleep(0.05)

# === Variáveis compartilhadas ===
shared_data = {
    "conv_output": np.zeros((IMG_SIZE - KERNEL_SIZE + 1, IMG_SIZE - KERNEL_SIZE + 1)),
    "kernel": kernel.copy()
}
lock = threading.Lock()

# === Pygame ===
pygame.init()
screen = pygame.display.set_mode((IMG_SIZE * 3, IMG_SIZE))
pygame.display.set_caption("Convolução Manual com Visualização")

def pygame_loop():
    running = True
    while running:
        screen.fill((30, 30, 30))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        with lock:
            conv_img = shared_data["conv_output"]
            kernel_img = shared_data["kernel"]

        orig_surface = to_pygame_surface(input_image, (IMG_SIZE, IMG_SIZE))
        conv_surface = to_pygame_surface(conv_img, (IMG_SIZE, IMG_SIZE))
        kernel_flat = kernel_img.reshape(KERNEL_SIZE, KERNEL_SIZE * 3)
        kernel_surface = to_pygame_surface(kernel_flat, (IMG_SIZE, IMG_SIZE))

        screen.blit(orig_surface, (0, 0))
        screen.blit(conv_surface, (IMG_SIZE, 0))
        screen.blit(kernel_surface, (IMG_SIZE * 2, 0))

        pygame.display.flip()
        time.sleep(0.03)

# === Iniciar Threads ===
train_thread = threading.Thread(target=train_loop)
pygame_thread = threading.Thread(target=pygame_loop)

train_thread.start()
pygame_thread.start()

train_thread.join()
pygame_thread.join()
