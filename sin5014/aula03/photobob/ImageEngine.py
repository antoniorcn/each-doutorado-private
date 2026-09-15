import math
from typing import Tuple
import pygame

class ImageEngine:
    def __init__(self, imagem) -> None:
        self.imagem = imagem
        self.imagem_cinza = pygame.transform.grayscale(imagem)
        self.vetor_h = []
        self.max_frequency = 0
        self.w, self.h = self.imagem_cinza.get_size()

    @staticmethod
    def pixel_to_gray( rgba : pygame.Color ) -> int:
        r, g, b, _ = rgba
        cor_pixel = min(255, int(math.sqrt( r**2 + b**2 + g**2 )))
        return cor_pixel

    def gerar_histograma(self):
        self.vetor_h = {}
        for y in range(self.h):
            for x in range(self.w):
                # print(self.imagem_cinza.get_at((x, y)))
                cor_pixel = ImageEngine.pixel_to_gray(self.imagem_cinza.get_at((x, y)))
                # print(self.imagem_cinza.get_at((x, y)), " -> ", cor_pixel)
                self.vetor_h[cor_pixel] = self.vetor_h.get(cor_pixel, 0) + 1
                # print("Frequencia: ", self.vetor_h[cor_pixel])
                if self.vetor_h[cor_pixel] > self.max_frequency:
                    self.max_frequency = self.vetor_h[cor_pixel]


    def mediana(self):
        total = sum(self.vetor_h)
        meio = total / 2
        acumulado = 0
        mediana = 0
        for intensidade, frequencia in enumerate(self.vetor_h):
            acumulado += frequencia
            if acumulado >= meio:
                mediana = intensidade
                break
        print(mediana)


    def mudar_brilho(self, brilho : int ) -> pygame.Surface :
        new_image = pygame.Surface( (self.w, self.h) )
        for y in range(self.h):
            for x in range(self.w):
                r, g, b, a = self.imagem.get_at((x, y))
                r += brilho
                g += brilho
                b += brilho
                r = max(min(r, 255), 0)
                g = max(min(g, 255), 0)
                b = max(min(b, 255), 0)
                # print(r, g, b, a)
                new_image.set_at((x, y), (r, g, b, a))
        return new_image

    def hist_draw(self, dimensoes : Tuple[int, int] = (512, 300)) -> pygame.Surface:
        hist_draw = pygame.Surface(dimensoes)
        print(hist_draw.get_size())
        for i, valor in enumerate(self.vetor_h):
            w = valor
            x1 = i * 2
            x2 = i * 2
            y1 = dimensoes[1] - w
            pygame.draw.line(hist_draw, (255, 255, 0), (x1, y1), (x2, y1 + w))
            # print(i, valor, x1, y1, x2, w)
        return hist_draw

    def equalizar(self):
        nova_imagem = pygame.Surface(self.imagem_cinza.get_size())
        w, h = self.imagem_cinza.get_size()
        total_pixels = w * h
        numero_ideal_pixels = int(total_pixels / 256)
        # Calcula a CDF
        cdf = [0] * 256
        somatoria = 0

        for i, valor in enumerate(self.vetor_h):
            somatoria += valor
            cdf[i] = somatoria

        dicionario_cor = {}
        for j in range(256):
            nova_cor = max(int((cdf[j] / numero_ideal_pixels)) - 1, 0)
            print(f"j: {j}   nova cor: {nova_cor}")
            dicionario_cor[j] = nova_cor

        somatoria = 0
        for y in range(h):
            for x in range(w):
                # cor_pixel, _, _ = self.imagem_cinza.get_at((x, y))
                cor_pixel = ImageEngine.pixel_to_gray(self.imagem_cinza.get_at((x, y)))
                nova_cor = dicionario_cor[cor_pixel]

                nova_imagem.set_at(
                    (x, y),
                    (nova_cor, nova_cor, nova_cor)
                )
        return nova_imagem
