import pygame
import math

# Tela
screen_width = 800
screen_height = 600
d = 500  # distância da câmera ao plano de projeção

# Lista de pontos 3D
points = [
    (100, 100, 300),
    (-100, 100, 300),
    (100, -100, 300),
    (0, 0, 500)
]

# Iniciar Pygame
pygame.init()
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()

running = True
while running:
    screen.fill((0, 0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    for x, y, z in points:
        # projeção simples
        x_proj = (x / z) * d
        y_proj = (y / z) * d

        # mapeia para tela
        screen_x = int(screen_width / 2 + x_proj)
        screen_y = int(screen_height / 2 - y_proj)

        pygame.draw.circle(screen, (255, 255, 255), (screen_x, screen_y), 5)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
