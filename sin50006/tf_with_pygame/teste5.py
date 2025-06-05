import pygame
import numpy as np
import math

# === Inicialização ===
pygame.init()
screen = pygame.display.set_mode((900, 700))
pygame.display.set_caption("Simulação 3D com Pygame")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 18)

# === Dados de exemplo ===
IMG_SIZE = 244
KERNEL_SIZE = 5
input_image = np.random.rand(IMG_SIZE, IMG_SIZE, 3)
conv_image = np.random.rand(IMG_SIZE, IMG_SIZE, 3)
kernel = np.random.randn(KERNEL_SIZE, KERNEL_SIZE, 3)
kernel_img = kernel.reshape(KERNEL_SIZE, KERNEL_SIZE * 3)
kernel_img = np.kron(kernel_img, np.ones((IMG_SIZE // KERNEL_SIZE, IMG_SIZE // (KERNEL_SIZE * 3))))

# === Converter para surface Pygame ===
def to_surface(img):
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    img = (img * 255).astype(np.uint8)
    surf = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
    return surf

# === Camadas ===
layers = [
    {"surface": to_surface(input_image), "z": 0, "label": "Imagem Original"},
    {"surface": to_surface(conv_image), "z": 100, "label": "Imagem Convolucionada"},
    {"surface": to_surface(kernel_img), "z": 200, "label": "Kernel"}
]

# === Estado da "câmera" ===
camera_angle_y = 0    # rotação horizontal (setas ← →)
camera_angle_x = 0    # rotação vertical (setas ↑ ↓)
camera_distance = 600 # distância simulada (zoom)

# === Projeção com rotação e zoom ===
def project_layer(surf, z, center=(450, 350)):
    global camera_angle_y, camera_angle_x, camera_distance
    angle_y_rad = math.radians(camera_angle_y)
    angle_x_rad = math.radians(camera_angle_x)

    # Coordenadas no espaço 3D
    x, y, z = 0, 0, z

    # Rotação em Y (esquerda/direita)
    x_rot = x * math.cos(angle_y_rad) - z * math.sin(angle_y_rad)
    z_rot = x * math.sin(angle_y_rad) + z * math.cos(angle_y_rad)

    # Rotação em X (cima/baixo)
    y_rot = y * math.cos(angle_x_rad) - z_rot * math.sin(angle_x_rad)
    z_final = y * math.sin(angle_x_rad) + z_rot * math.cos(angle_x_rad)

    # Perspectiva
    depth = 1 / (1 + z_final / camera_distance)
    size = int(surf.get_width() * depth), int(surf.get_height() * depth)
    scaled = pygame.transform.smoothscale(surf, size)
    pos = (center[0] - size[0] // 2 + int(x_rot), center[1] - size[1] // 2 + int(y_rot))

    return scaled, pos, depth

# === Loop principal ===
running = True
while running:
    screen.fill((10, 10, 10))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Controles
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        camera_angle_y -= 2
    if keys[pygame.K_RIGHT]:
        camera_angle_y += 2
    if keys[pygame.K_UP]:
        camera_angle_x -= 2
    if keys[pygame.K_DOWN]:
        camera_angle_x += 2
    if keys[pygame.K_MINUS]:
        camera_distance += 10
    if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]:
        camera_distance = max(100, camera_distance - 10)

    # Desenhar camadas (traseira primeiro)
    for layer in sorted(layers, key=lambda l: -l["z"]):
        surface, pos, depth = project_layer(layer["surface"], layer["z"])
        alpha = int(255 * depth)
        if alpha < 255:
            surface.set_alpha(alpha)
        screen.blit(surface, pos)

        label = font.render(layer["label"], True, (255, 255, 255))
        screen.blit(label, (pos[0], pos[1] - 20))

    # Instruções
    help_text = [
        "←/→: Girar",
        "↑/↓: Inclinar",
        "+/-: Zoom",
        "ESC: Sair"
    ]
    for i, txt in enumerate(help_text):
        t = font.render(txt, True, (180, 180, 180))
        screen.blit(t, (10, 10 + i * 20))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
