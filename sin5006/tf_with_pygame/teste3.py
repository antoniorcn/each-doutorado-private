import numpy as np
import pygame
import threading
import time

# === Parâmetros ===
IMG_SIZE = 244
KERNEL_SIZE = 5
LEARNING_RATE = 0.0005

# === Gerar imagem de entrada (RGB) e imagem alvo (escala de cinza aleatória) ===
input_image = np.random.rand(IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
target_output = np.random.rand(IMG_SIZE - KERNEL_SIZE + 1, IMG_SIZE - KERNEL_SIZE + 1).astype(np.float32)

# === Inicializar kernel (filtro 5x5x3) ===
kernel = np.random.randn(KERNEL_SIZE, KERNEL_SIZE, 3).astype(np.float32)

# === Variáveis compartilhadas ===
shared_data = {
    "conv_output": np.zeros_like(target_output),
    "kernel": kernel.copy()
}
lock = threading.Lock()

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

# === Treinamento passo a passo ===
def train_loop():
    global kernel
    h_out, w_out = target_output.shape

    for epoch in range(1):  # rodamos uma vez, pixel por pixel
        for i in range(h_out):
            for j in range(w_out):
                # Extrai região e calcula convolução
                region = input_image[i:i+KERNEL_SIZE, j:j+KERNEL_SIZE, :]
                value = np.sum(region * kernel)
                error = target_output[i, j] - value
                loss = error ** 2

                # Atualiza a saída
                with lock:
                    shared_data["conv_output"][i, j] = value

                # Atualiza kernel (pseudo gradiente)
                for ki in range(KERNEL_SIZE):
                    for kj in range(KERNEL_SIZE):
                        for c in range(3):
                            grad = -2 * error * region[ki, kj, c]
                            kernel[ki, kj, c] -= LEARNING_RATE * grad

                # Atualiza kernel compartilhado
                with lock:
                    shared_data["kernel"] = kernel.copy()

                # Permite visualização após cada passo
                pygame.event.pump()
                # time.sleep(0.002)  # ajuste para visualização

# === Pygame ===
pygame.init()
screen = pygame.display.set_mode((IMG_SIZE * 3, IMG_SIZE))
pygame.display.set_caption("Convolução Manual Passo a Passo")

def pygame_loop():
    running = True
    clock = pygame.time.Clock()

    while running:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        with lock:
            screen.fill((0, 0, 0))
            
            conv_img = shared_data["conv_output"].copy()
            kernel_img = shared_data["kernel"].copy()

            orig_surface = to_pygame_surface(input_image, (IMG_SIZE, IMG_SIZE))
            conv_surface = to_pygame_surface(conv_img, (IMG_SIZE, IMG_SIZE))
            kernel_flat = kernel_img.reshape(KERNEL_SIZE, KERNEL_SIZE * 3)
            kernel_surface = to_pygame_surface(kernel_flat, (IMG_SIZE, IMG_SIZE))

            screen.blit(orig_surface, (0, 0))
            screen.blit(conv_surface, (IMG_SIZE, 0))
            screen.blit(kernel_surface, (IMG_SIZE * 2, 0))

            pygame.display.flip()
        clock.tick(60)

# === Iniciar Threads ===
train_thread = threading.Thread(target=train_loop)
pygame_thread = threading.Thread(target=pygame_loop)

train_thread.start()
pygame_thread.start()

train_thread.join()
pygame_thread.join()
