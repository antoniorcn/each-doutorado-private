import pygame
import numpy as np
import cv2
import math
import time
import sys

# === CONFIGURAÇÕES ===
IMG_PATH = 'sua_imagem.png'  # Substitua pela sua imagem
KERNEL = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]], dtype=np.float32)
DELAY = 0.005  # Delay entre passos
IMG_SIZE = 244

# === PROJEÇÃO 3D ===
def projetar(x, y, z, ang_x, ang_y, offset_x, offset_y, scale=1.0):
    cosx, sinx = math.cos(ang_x), math.sin(ang_x)
    cosy, siny = math.cos(ang_y), math.sin(ang_y)
    y2 = y * cosx - z * sinx
    z2 = y * sinx + z * cosx
    x2 = x * cosy + z2 * siny
    return int(x2 * scale + offset_x), int(y2 * scale + offset_y)

# === DESENHAR LÂMINA USANDO BLIT PROJETADO ===
def desenhar_lamina(tela, imagem, z, ang_x, ang_y, offset_x, offset_y):
    h, w = imagem.shape[:2]
    surface = pygame.surfarray.make_surface(imagem.swapaxes(0, 1))
    px, py = projetar(0, 0, z, ang_x, ang_y, offset_x, offset_y)
    tela.blit(surface, (px - w // 2, py - h // 2))


def desenhar_kernel(tela, pos_x, pos_y, kernel_size, z, ang_x, ang_y, offset_x, offset_y):
    for y in range(kernel_size):
        for x in range(kernel_size):
            px, py = projetar((pos_x + x) - IMG_SIZE//2, (pos_y + y) - IMG_SIZE//2, z,
                              ang_x, ang_y, offset_x, offset_y)
            pygame.draw.rect(tela, (0, 255, 255), (px, py, 3, 3))

# === CONVOLUÇÃO RGB POR LINHA ===
def convolucao_por_linha(img_rgb, kernel):
    h, w, _ = img_rgb.shape
    k = kernel.shape[0]
    offset = k // 2
    resultado = np.zeros_like(img_rgb)
    linhas_intermediarias = []

    for y in range(offset, h - offset):
        for x in range(offset, w - offset):
            for c in range(3):  # RGB
                regiao = img_rgb[y - offset:y + offset + 1, x - offset:x + offset + 1, c]
                valor = np.sum(regiao * kernel)
                resultado[y, x, c] = np.clip(valor, 0, 255)
        linhas_intermediarias.append(resultado.copy())
        yield resultado.copy(), y, linhas_intermediarias

# === MAIN ===
def visualizar_convolucao(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pygame.init()
    screen = pygame.display.set_mode((1000, 800))
    pygame.display.set_caption("Visualização 3D do Processo de Convolução")
    clock = pygame.time.Clock()

    offset_x, offset_y = 500, 400
    ang_x, ang_y = math.radians(30), math.radians(30)
    dragging = False
    paused = False
    last_mouse = None

    gerador = convolucao_por_linha(img_rgb, KERNEL)
    resultado_final = None
    linha_atual = 0
    intermediarias = []

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 2:
                dragging = True
                last_mouse = pygame.mouse.get_pos()
            elif e.type == pygame.MOUSEBUTTONUP and e.button == 2:
                dragging = False
            elif e.type == pygame.MOUSEMOTION and dragging:
                mx, my = pygame.mouse.get_pos()
                dx, dy = mx - last_mouse[0], my - last_mouse[1]
                ang_y += dx * 0.01
                ang_x += dy * 0.01
                last_mouse = (mx, my)
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_r:
                    gerador = convolucao_por_linha(img_rgb, KERNEL)
                    resultado_final = None
                    linha_atual = 0
                    intermediarias = []
                    paused = False

        if not paused and resultado_final is None:
            try:
                resultado_final, linha_atual, intermediarias = next(gerador)
            except StopIteration:
                pass

        screen.fill((30, 30, 30))

        # Camada final (fundo claro)
        if resultado_final is not None:
            desenhar_lamina(screen, resultado_final, z=+30, ang_x=ang_x, ang_y=ang_y,
                            offset_x=offset_x, offset_y=offset_y)

        # Camadas intermediárias (a cada linha)
        z_start = -30
        for i, img_inter in enumerate(intermediarias[-10:]):
            z_layer = z_start + i * 6
            desenhar_lamina(screen, img_inter, z=z_layer, ang_x=ang_x, ang_y=ang_y,
                            offset_x=offset_x, offset_y=offset_y)

        # Camada da imagem original (mais próxima)
        desenhar_lamina(screen, img_rgb, z=-60, ang_x=ang_x, ang_y=ang_y,
                        offset_x=offset_x, offset_y=offset_y)

        # Camada do kernel (mais distante)
        desenhar_kernel(screen, pos_x=KERNEL.shape[0]//2, pos_y=linha_atual,
                        kernel_size=KERNEL.shape[0], z=-90,
                        ang_x=ang_x, ang_y=ang_y,
                        offset_x=offset_x, offset_y=offset_y)
        desenhar_lamina(screen, img_rgb, z=-60, ang_x=ang_x, ang_y=ang_y,
                        offset_x=offset_x, offset_y=offset_y)

        

        # Camadas intermediárias (a cada linha)
        z_start = -30
        for i, img_inter in enumerate(intermediarias[-10:]):
            z_layer = z_start + i * 6
            desenhar_lamina(screen, img_inter, z=z_layer, ang_x=ang_x, ang_y=ang_y,
                            offset_x=offset_x, offset_y=offset_y)

        # Camada final (fundo claro)
        if resultado_final is not None:
            desenhar_lamina(screen, resultado_final, z=+30, ang_x=ang_x, ang_y=ang_y,
                            offset_x=offset_x, offset_y=offset_y)

        # Botões e instruções
        fonte = pygame.font.SysFont(None, 24)
        texto = "Espaço: Play/Pause | R: Reset"
        txt_surface = fonte.render(texto, True, (255, 255, 255))
        screen.blit(txt_surface, (20, 20))

        pygame.display.flip()
        time.sleep(DELAY)
        clock.tick(60)

visualizar_convolucao(IMG_PATH)
