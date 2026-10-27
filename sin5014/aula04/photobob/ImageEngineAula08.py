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

    @staticmethod
    def pintar_bordas(imagem : pygame.Surface, mapa : List[List[int]],
                      ponto : Tuple[int, int], indice_objeto : List[int]):
        x, y = ponto
        cor_ponto = imagem.get_at(ponto)[0]
        w, h = imagem.get_size()
        vizinhos = ImageEngine.verificar_vizinhanca(imagem, ponto, 1)
        lista_vizinhos_a_verificar = []
        tem_vizinho_no_fundo_branco = False
        for linha in range(-1, 2):
            for coluna in range(-1, 2):
                n_x, n_y = (x + coluna, y + linha)
                if n_x >= 0 and n_x < w and n_y > 0 and n_y < h:
                    vizinho = vizinhos[linha, coluna]
                    if vizinho >= 0:
                        if vizinho==255 and mapa[n_x][n_y] == -1:
                            tem_vizinho_no_fundo_branco = True
                            mapa[n_x][n_y] = 255
        mapa[x][y] = indice_objeto[0] if tem_vizinho_no_fundo_branco else 0
        print(f"Verificando vizinhos ao ponto {ponto}:{cor_ponto} => {lista_vizinhos_a_verificar}")


    @staticmethod
    def transformar_mapa_para_imagem( mapa : List[List[int]] ) -> pygame.Surface:
        h = len(mapa)
        w = len(mapa[0])
        nova_imagem = pygame.Surface((w, h), 0, 32)
        for y in range(h):
            for x in range(w):
                valor = mapa[x][y]
                cor_pixel = (0, 0, 0)
                if valor == 255:
                    cor_pixel = (255, 255, 255)
                elif valor != 0:
                    cor_pixel = (255, 0, 0)
                nova_imagem.set_at((x, y), cor_pixel )
        return nova_imagem

    @staticmethod
    def detectar_bordas(imagem : pygame.Surface):
        w, h = imagem.get_size()
        mapa = [[ -1 for _ in range(h)] for _ in range(w)]
        for y in range(h):
            for x in range(w):
                pixel_gray = imagem.get_at((x, y))[0]
                if mapa[x][y] == -1:
                    if pixel_gray == 255:
                        mapa[x][y] = 255
                    else:
                        ImageEngine.pintar_bordas(imagem, mapa, (x, y), [1])
        return mapa

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
