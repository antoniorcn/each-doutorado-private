import pygame
import math

# Configurações da tela
screen_width = 800
screen_height = 600
d = 500  # distância da câmera para o plano de projeção

# Geração dos pontos da esfera
def generate_sphere_points(radius, num_points):
    points = []
    golden_angle = math.pi * (3 - math.sqrt(5))  # distribuição tipo Fibonacci

    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2  # vai de 1 a -1
        radius_at_y = math.sqrt(1 - y * y)
        theta = golden_angle * i

        x = math.cos(theta) * radius_at_y
        z = math.sin(theta) * radius_at_y

        points.append((radius * x, radius * y, radius * z))
    
    return points

points = generate_sphere_points(radius=150, num_points=50)

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
        # Projeção perspectiva
        x_proj = (x / (z + 400)) * d
        y_proj = (y / (z + 400)) * d

        screen_x = int(screen_width / 2 + x_proj)
        screen_y = int(screen_height / 2 - y_proj)

        pygame.draw.circle(screen, (255, 255, 255), (screen_x, screen_y), 4)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
