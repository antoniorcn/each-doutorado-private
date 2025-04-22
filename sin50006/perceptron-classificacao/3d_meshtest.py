import pygame
import math
import random

# Inicialização
pygame.init()
WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Cores
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
RED = (255, 100, 100)

# Parâmetros
escala_erro = 1.0
offset_x, offset_y = 0, 0
angle_x, angle_y = math.radians(30), math.radians(45)
mouse_drag_mode = None  # "rotate" ou "pan"
last_mouse_pos = None

# Histórico de erro
historico_erro = []

def erro(x, w):
    return math.sin(x) * math.cos(w) * escala_erro

def rotacionar_ponto(x, y, z, ax, ay):
    cos_y, sin_y = math.cos(ay), math.sin(ay)
    xz = x * cos_y + z * sin_y
    zz = -x * sin_y + z * cos_y

    cos_x, sin_x = math.cos(ax), math.sin(ax)
    yz = y * cos_x - zz * sin_x
    zz = y * sin_x + zz * cos_x

    return xz, yz, zz

def project_point(x, y, z):
    scale = 20
    x, y, z = rotacionar_ponto(x, y, z, angle_x, angle_y)
    px = int(WIDTH / 2 + x * scale + offset_x)
    py = int(HEIGHT / 2 + y * scale + offset_y)
    return px, py

def gerar_malha():
    malha = []
    x_range = w_range = range(-5, 6)
    for x in x_range:
        linha = []
        for w in w_range:
            z = erro(x, w)
            px, py = project_point(x, w, z)
            linha.append((px, py))
        malha.append(linha)
    return malha

def desenhar_grafico_erro(surface, historico):
    graf_x = 50
    graf_y = HEIGHT - 200
    graf_w = WIDTH - 100
    graf_h = 150
    pygame.draw.rect(surface, GRAY, (graf_x, graf_y, graf_w, graf_h), 1)

    if len(historico) > 1:
        max_erro = max(historico)
        min_erro = min(historico)
        range_erro = max_erro - min_erro or 1
        pontos = [
            (
                graf_x + int(i * graf_w / len(historico)),
                graf_y + graf_h - int((e - min_erro) * graf_h / range_erro)
            )
            for i, e in enumerate(historico)
        ]
        pygame.draw.lines(surface, RED, False, pontos, 2)

# Loop principal
running = True
frame_contador = 0

while running:
    screen.fill(BLACK)

    # Simular novo valor de erro
    if frame_contador % 30 == 0:
        erro_simulado = random.uniform(0.5, 1.5) * abs(math.sin(pygame.time.get_ticks() * 0.001))
        historico_erro.append(erro_simulado)
        if len(historico_erro) > 300:
            historico_erro.pop(0)

    malha = gerar_malha()

    # Desenhar malha
    for linha in malha:
        pygame.draw.lines(screen, GREEN, False, linha, 1)
    for i in range(len(malha[0])):
        coluna = [linha[i] for linha in malha]
        pygame.draw.lines(screen, GREEN, False, coluna, 1)

    # Desenhar gráfico
    desenhar_grafico_erro(screen, historico_erro)

    pygame.display.flip()
    clock.tick(60)
    frame_contador += 1

    # Eventos
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                escala_erro += 0.1
            elif event.key == pygame.K_DOWN:
                escala_erro -= 0.1

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Botão esquerdo: rotação
                mouse_drag_mode = "rotate"
                last_mouse_pos = pygame.mouse.get_pos()
            elif event.button == 3:  # Botão direito: pan
                mouse_drag_mode = "pan"
                last_mouse_pos = pygame.mouse.get_pos()

        elif event.type == pygame.MOUSEBUTTONUP:
            mouse_drag_mode = None

        elif event.type == pygame.MOUSEMOTION and mouse_drag_mode:
            mx, my = pygame.mouse.get_pos()
            lx, ly = last_mouse_pos
            dx, dy = mx - lx, my - ly

            if mouse_drag_mode == "rotate":
                angle_y += dx * 0.01  # rotação horizontal
                angle_x += dy * 0.01  # rotação vertical

            elif mouse_drag_mode == "pan":
                offset_x += dx
                offset_y += dy

            last_mouse_pos = (mx, my)

pygame.quit()
