import os
import sys
import math
import pygame
import numpy as np
from PIL import Image
from pygame import Surface
from typing import Dict, List, Tuple


class Convolucao3DAnimation():

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Visualizacao 3D - Convolucao Manual")
        self.clock = pygame.time.Clock()

        # Câmera
        self.camera_pos = [0, -300, -1000]
        self.yaw = math.radians(-20)
        self.pitch = math.radians(15)
        self.fov = 500

        self.dragging = False
        self.rotating = False
        self.last_mouse = (0, 0)

        self.imagens = []
        self.elementos = []

    def rotate_yaw_pitch(self, point):
        x, y, z = point
        x1 = x * math.cos(self.yaw) - z * math.sin(self.yaw)
        z1 = x * math.sin(self.yaw) + z * math.cos(self.yaw)
        y2 = y * math.cos(self.pitch) - z1 * math.sin(self.pitch)
        z2 = y * math.sin(self.pitch) + z1 * math.cos(self.pitch)
        return [x1, y2, z2]

    def project(self, x, y, z):
        relative = [x - self.camera_pos[0], y - self.camera_pos[1], z - self.camera_pos[2]]
        rotated = self.rotate_yaw_pitch(relative)
        if rotated[2] <= 1:
            return None
        local_scale = self.fov / rotated[2]
        local_x_proj = int(400 + rotated[0] * local_scale)
        local_y_proj = int(300 - rotated[1] * local_scale)
        return local_x_proj, local_y_proj, local_scale

    @staticmethod
    def processar_imagem(image_path: str, tamanho: Tuple[int, int] = (244, 244)) -> np.ndarray:
        imagem = Image.open(image_path).convert("RGB").resize(tamanho)
        pixels = np.array(imagem, dtype=np.float32) / 255.0
        return pixels

    def carregar_imagens(self, pasta='images'):
        for nome in sorted(os.listdir(pasta)):
            if nome.lower().endswith(('.jpg', '.png')):
                img_path = os.path.join(pasta, nome)
                pixels = self.processar_imagem(img_path)
                self.imagens.append(pixels)
        return self.imagens

    @staticmethod
    def generate_surface_from_nparray(pixels):
        surface = pygame.surfarray.make_surface(np.transpose((pixels * 255).astype(np.uint8), (1, 0, 2)))
        return pygame.transform.scale(surface, (200, 150))

    def inicializar_cena(self):
        z = 0
        for img in self.imagens:
            self.elementos.append({
                'tipo': 'quadro',
                'img': self.generate_surface_from_nparray(img),
                'pos': [0, 0, z]
            })
            z += 300

    def aplicar_convolucao_manual(self, imagem: np.ndarray, kernel: np.ndarray):
        altura, largura, canais = imagem.shape
        k = kernel.shape[0]
        pad = k // 2
        saida = np.zeros_like(imagem)
        imagem_pad = np.pad(imagem, ((pad, pad), (pad, pad), (0, 0)), mode='constant')

        for y in range(altura):
            for x in range(largura):
                for c in range(canais):
                    regiao = imagem_pad[y:y + k, x:x + k, c]
                    saida[y, x, c] = np.clip(np.sum(regiao * kernel), 0, 1)
        return saida

    def draw(self):
        self.screen.fill((25, 25, 40))
        elementos_render = []

        for elem in self.elementos:
            proj = self.project(*elem['pos'])
            if proj:
                elementos_render.append((proj[2], elem, proj))

        elementos_render.sort(reverse=False)

        for _, elem, proj in elementos_render:
            x_proj, y_proj, scale = proj
            img_scaled = pygame.transform.scale(elem['img'], (
                int(200 * scale), int(150 * scale)))
            self.screen.blit(img_scaled, (x_proj - img_scaled.get_width() // 2, y_proj - img_scaled.get_height() // 2))

        pygame.display.flip()
        self.clock.tick(60)

    def execute(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.dragging = True
                    elif event.button == 3:
                        self.rotating = True
                    self.last_mouse = pygame.mouse.get_pos()
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.dragging = False
                    elif event.button == 3:
                        self.rotating = False
                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging or self.rotating:
                        mx, my = event.pos
                        dx = mx - self.last_mouse[0]
                        dy = my - self.last_mouse[1]
                        self.last_mouse = (mx, my)
                        if self.dragging:
                            self.camera_pos[0] -= dx * 1.5
                            self.camera_pos[1] += dy * 1.5
                        elif self.rotating:
                            self.yaw += dx * 0.01
                            self.pitch += dy * 0.01
                            self.pitch = max(-math.pi / 2 + 0.1, min(math.pi / 2 - 0.1, self.pitch))

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                self.camera_pos[2] += 20
            if keys[pygame.K_s]:
                self.camera_pos[2] -= 20

            self.draw()

        pygame.quit()


# Executar
app = Convolucao3DAnimation()
input_images = app.carregar_imagens()

# Convoluir e empilhar
kernel = np.ones((5, 5)) / 25.0
img_base = input_images[0]
img_conv = app.aplicar_convolucao_manual(img_base, kernel)

app.imagens = [img_base, img_conv]
app.inicializar_cena()
app.execute()
