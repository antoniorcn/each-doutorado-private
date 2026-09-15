import pygame
from math import cos, sin, atan, acos, pi, atan2
import numpy as np

tela = pygame.display.set_mode((800, 600), 0, 32)

triangulo = np.array([ (400, 50, 1), (600, 200, 1), (200, 200, 1) ])

executando = True

while executando:
    
    # Regras
    ponto1 = tuple(triangulo[2])
    ponto2 = tuple(triangulo[1])
    matriz_translacao = np.array([
        (1, 0, -ponto1[0]),
        (0, 1, -ponto1[1]),
        (0, 0, 1),
    ])

    #                            dY    /  dX
    # tangente = (ponto2[1] - ponto1[1]) / (ponto2[0] - ponto1[0])
    angulo = atan2(ponto2[0] - ponto1[0], ponto2[1] - ponto1[1])
    # angulo = pi / 2

    # dx = ponto2[0] - ponto1[0]
    # dy = ponto2[1] - ponto2[1]
    # h = ((dx ** 2) + (dy ** 2)) ** (1/2)
    # relacao_dx_h = dx / h
    # angulo = acos( relacao_dx_h )

    matriz_rotacao = np.array( [
        (cos(angulo), -sin(angulo), 0),
        (sin(angulo), cos(angulo), 0),
        (0, 0, 1)
    ])

    matriz_translacao_retorno = np.array([
        (1, 0, 0),
        (0, 1, +ponto1[1]),
        (0, 0, 1),
    ])

    matriz_transformacao = (matriz_translacao_retorno @ matriz_rotacao) @ matriz_translacao

    triangulo_original = [tuple(p[0:2]) for p in triangulo]
    # print(triangulo_original)
    novo_triangulo =  []
    for ponto in triangulo:
        #               3 x 3              1 x 3
        ponto_temp = matriz_transformacao @ ponto
        novo_triangulo.append( tuple(ponto_temp[:2]) )
    # print(f"P1:{ponto1}  P2:{ponto2}  Ang: {angulo}  dX: {dx} dY: {dy}  h:{h} Rel Adj/H: {relacao_dx_h}")
    

    # Desenha tela
    pygame.draw.line(tela, (255, 255, 0), (0, 0), (0, 600), 2)
    pygame.draw.line(tela, (255, 255, 0), (0, 600), (800, 600), 2)

    pygame.draw.polygon( tela, (255, 0, 0), triangulo_original, 2)
    pygame.draw.polygon( tela, (255, 255, 0), novo_triangulo, 2)

    pygame.display.update()

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            executando = False