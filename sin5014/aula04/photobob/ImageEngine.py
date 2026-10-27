import math
from typing import List, Tuple
from enum import Enum
import pygame
import numpy as np
import numpy.typing as npt



class Direcao(Enum):
    NENHUMA = "nenhuma"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    DIAGONAL_ESQUERDA = "diagonal_esquerda"
    DIAGONAL_DIREITA = "diagonal_direita"

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

    def preenchimento_recursivo(self, imagem : pygame.Surface, semente : Tuple[int, int]):
        pass

    @staticmethod
    def verificar_vizinhanca(imagem : pygame.Surface,
                             pos : Tuple[int, int], neighbor_size=1
                             ) -> npt.NDArray[np.float64]:
        matrix_size = (neighbor_size * 2) + 1
        vizinhos = np.ones((matrix_size, matrix_size)) * -1
        w, h = imagem.get_size()
        x, y = pos
        y1 = y - neighbor_size
        x1 = x - neighbor_size
        for j in range(matrix_size):
            for i in range(matrix_size):
                vy = y1 + j
                vx = x1 + i
                if 0 < vy < h and 0 < vx < w:
                    vizinhos[j, i] = imagem.get_at((vx, vy))[0]
        return vizinhos

    # @staticmethod
    # def pinta_mapa_vizinhanca(imagem : pygame.Surface, mapa : List[List[int]],
    #                           ponto : Tuple[int, int], indice_objeto : List[int],
    #                           passagem_cores : List[Tuple[int, int]]):
    #     x, y = ponto
    #     cor_ponto = imagem.get_at(ponto)[0]
    #     w, h = imagem.get_size()
    #     vizinhos = ImageEngine.verificar_vizinhanca(imagem, ponto, 1)
    #     lista_vizinhos_a_verificar = []
    #     for linha in range(-1, 2):
    #         for coluna in range(-1, 2):
    #             n_x, n_y = (x + coluna, y + linha)
    #             if n_x >= 0 and n_x < w and n_y > 0 and n_y < h:
    #                 vizinho = vizinhos[linha, coluna]
    #                 if vizinho >= 0:
    #                     cor_vizinho = imagem.get_at((n_x, n_y))[0]
    #                     objeto_mapa = mapa[n_x][n_y]
    #                     # mapa[n_x][n_y] = 0 if vizinho == 255 else indice_objeto[0]
    #                     if vizinho!=255 and objeto_mapa == -1:
    #                         # print(f"Vizinhos ({coluna}, {linha}):{vizinho}  "
    #                         #       f"Cor: {ponto}:{cor_ponto}  "
    #                         #       f"Mapa: ({n_x}, {n_y}):{objeto_mapa}  "
    #                         #       f"Cor Vizinho: ({n_x}, {n_y}):{cor_vizinho}")
    #                         mapa[n_x][n_y] = indice_objeto[0]
    #                         nova_pos = (n_x, n_y)
    #                         lista_vizinhos_a_verificar.append( nova_pos )
    #                     mapa[n_x][n_y] = 0 if vizinho == 255 else indice_objeto[0]
    #                     # else:
    #                     #     if cor_vizinho != cor_ponto and\
    #                     #         (cor_vizinho, cor_ponto) not in passagem_cores:
    #                     #         passagem_cores.extend(
    #                     #             [(cor_vizinho, cor_ponto), (cor_ponto, cor_vizinho)]
    #                     #             )
    #                     #         indice_objeto[0] = indice_objeto[0] + 1
    #     print(f"Verificando vizinhos ao ponto {ponto}:{cor_ponto} => {lista_vizinhos_a_verificar}")
    #     # if len(lista_vizinhos_a_verificar) > 0:
    #     #     vizinho = lista_vizinhos_a_verificar.pop()
    #     for vizinho in lista_vizinhos_a_verificar:
    #         ImageEngine.pinta_mapa_vizinhanca(imagem, mapa, vizinho, indice_objeto, passagem_cores)

    @staticmethod
    def pinta_mapa_vizinhanca(imagem : pygame.Surface, mapa : List[List[int]],
                              ponto : Tuple[int, int], indice_objeto : List[int],
                              passagem_cores : List[Tuple[int, int]]):
        x, y = ponto
        cor_ponto = imagem.get_at(ponto)[0]
        w, h = imagem.get_size()
        vizinhos = ImageEngine.verificar_vizinhanca(imagem, ponto, 1)
        lista_vizinhos_a_verificar = []
        for linha in range(-1, 2):
            for coluna in range(-1, 2):
                n_x, n_y = (x + coluna, y + linha)
                if n_x >= 0 and n_x < w and n_y > 0 and n_y < h:
                    vizinho = vizinhos[linha, coluna]
                    if vizinho >= 0:
                        cor_vizinho = imagem.get_at((n_x, n_y))[0]
                        if vizinho!=255 and mapa[n_x][n_y] == -1:
                            mapa[n_x][n_y] = indice_objeto[0]
                            nova_pos = (n_x, n_y)
                            lista_vizinhos_a_verificar.append( nova_pos )
                        mapa[n_x][n_y] = 0 if vizinho == 255 else indice_objeto[0]
        print(f"Verificando vizinhos ao ponto {ponto}:{cor_ponto} => {lista_vizinhos_a_verificar}")
        for vizinho in lista_vizinhos_a_verificar:
            ImageEngine.pinta_mapa_vizinhanca(imagem, mapa, vizinho, indice_objeto, passagem_cores)

    @staticmethod
    def identificar_objetos(imagem : pygame.Surface) -> int:
        count_objetos = [0]
        w, h = imagem.get_size()
        mapa = [[ -1 for _ in range(h)] for _ in range(w)]
        passagem_cores = []
        for y in range(h):
            for x in range(w):
                pixel_gray = imagem.get_at((x, y))[0]
                if mapa[x][y] == -1:
                    if pixel_gray == 255:
                        mapa[x][y] = 0
                    else:
                        print("Novo objeto detectado")
                        count_objetos[0] = count_objetos[0] + 1
                        ImageEngine.pinta_mapa_vizinhanca(imagem, mapa, (x, y),
                                                          count_objetos, passagem_cores)
        # print(mapa)
        return count_objetos[0]

    # def preenchimento_djistra(self, imagem : pygame.Surface,
    #                           semente : Tuple[int, int]) -> pygame.Surface:
    #     img_cinza = pygame.transform.grayscale(imagem)
    #     w, h = imagem.get_size()
    #     nova_imagem = pygame.Surface((w, h))
    #     for y in range(1, h):
    #         for x in range(1, w):
    #             # verificar as vizinhanças
    #             vizinhos = np.array((neighbor_size, neighbor_size))
    #             vizinhos[0, 0] = ImageEngine.pixel_to_gray(imagem.get_at((x - 1, y - 1)))
    #             vizinhos[0, 1] = ImageEngine.pixel_to_gray(imagem.get_at((x, y - 1)))
    #             vizinhos[0, 2] = ImageEngine.pixel_to_gray(imagem.get_at((x + 1, y - 1)))
    #             vizinhos[1, 0] = ImageEngine.pixel_to_gray(imagem.get_at((x - 1, y)))
    #             vizinhos[1, 1] = ImageEngine.pixel_to_gray(imagem.get_at((x, y)))
    #             vizinhos[1, 2] = ImageEngine.pixel_to_gray(imagem.get_at((x + 1, y)))
    #             vizinhos[2, 0] = ImageEngine.pixel_to_gray(imagem.get_at((x - 1, y + 1)))
    #             vizinhos[2, 1] = ImageEngine.pixel_to_gray(imagem.get_at((x, y + 1)))
    #             vizinhos[2, 2] = ImageEngine.pixel_to_gray(imagem.get_at((x + 1, y + 1)))
    #             nova_image.set_at()

    # def convolucao(self, imagem : pygame.Surface, mascaras : List[np.array]) -> np.array:
    #     mask_size = mascaras[0].shape
    #     w, h = imagem.get_size()
    #     np.zeros((w))
    #     pass


    def detector_linhas_por_soma(self, imagem : pygame.Surface) -> Direcao:
        w, h = imagem.get_size()
        soma_horizontal = 0
        soma_vertical = 0
        soma_diagonal_esquerda = 0
        soma_diagonal_direita = 0
        for y in range(h):
            for x in range(w):
                gray_pixel = ImageEngine.pixel_to_gray(self.imagem_cinza.get_at((x, y)))

        threshold = 0

        if soma_horizontal > threshold:
            return Direcao.HORIZONTAL
        elif soma_vertical > threshold:
            return Direcao.VERTICAL
        elif soma_diagonal_esquerda > threshold:
            return Direcao.DIAGONAL_ESQUERDA
        elif soma_diagonal_direita > threshold:
            return Direcao.DIAGONAL_DIREITA
        return Direcao.NENHUMA


    # def detector_linhas_por_quadrante(self, imagem : pygame.Surface) -> Direcao:
    #     w, h = imagem.get_size()
    #     soma_horizontal = 0
    #     soma_vertical = 0
    #     soma_diagonal_esquerda = 0
    #     soma_diagonal_direita = 0
    #     for y in range(h):
    #         for x in range(w):
    #             gray_pixel = ImageEngine.pixel_to_gray(self.imagem_cinza.get_at((x, y)))
    #             if gray_pixel == pygame.Color(0, 0, 0):
    #                 if soma_horizontal > threshold:
    #                     return Direcao.HORIZONTAL
    #                 elif soma_vertical > threshold:
    #                     return Direcao.VERTICAL
    #                 elif soma_diagonal_esquerda > threshold:
    #                     return Direcao.DIAGONAL_ESQUERDA
    #                 elif soma_diagonal_direita > threshold:
    #                     return Direcao.DIAGONAL_DIREITA
    #                 return Direcao.NENHUMA

    #     threshold = 0
