from math import cos, sin
from typing import Tuple
import numpy as np
import pygame


pygame.init()


cubo = [
    # Quadro da frente
    ((10, 10, 10), (50, 10, 10)), 
    ((50, 10, 10), (50, 50, 10)),
    ((50, 50, 10), (10, 50, 10)),
    ((10, 50, 10), (10, 10, 10)),

    # Quadro Fundo
    ((10, 10, 50), (50, 10, 50)),
    ((50, 10, 50), (50, 50, 50)),
    ((50, 50, 50), (10, 50, 50)),
    ((10, 50, 50), (10, 10, 50)),

    # Ligações entre frente e fundo
    ((10, 10, 10), (10, 10, 50)), 
    ((50, 10, 10), (50, 10, 50)),
    ((50, 50, 10), (50, 50, 50)),
    ((10, 50, 10), (10, 50, 50)),
]

universo = [ np.array(cubo) ]

YELLOW = (255, 255, 0)

tela = pygame.display.set_mode( (800, 600), 0, 32)

def projetar_ponto( vertice3d : np.ndarray, distancia : float ) -> Tuple:
    X, Y, Z, _ = vertice3d
    pX = (X * distancia) / Z
    pY = (Y * distancia) / Z
    return pX, pY

distancia = 10
incremento_distancia = 0

angulo_x = 0
angulo_y = 0
angulo_z = 0
incremento_angulo_x = 0
incremento_angulo_y = 0
incremento_angulo_z = 0

executando = True
while executando:

    # Calcular
    distancia += incremento_distancia
    angulo_x += incremento_angulo_x
    angulo_y += incremento_angulo_y
    angulo_z += incremento_angulo_z

    # matriz_rotacao = [
    #     (cos(angulo_y) * cos(angulo_z), -cos(angulo_y) * sin(angulo_z), sin(angulo_y), 0),
    #     ( sin(angulo_x) * sin(angulo_y) * cos(angulo_z) + cos(angulo_x)*sin(angulo_z),  )
    # ]

    matriz_rotacao_x = np.array([
        ( 1, 0, 0, 0),
        (0, cos(angulo_x), -sin(angulo_x), 0),
        (0, sin(angulo_x), cos(angulo_x), 0),
        (0, 0, 0, 1)
    ])

    matriz_rotacao_y = np.array([
        (cos(angulo_y), 0, sin(angulo_y), 0),
        (0, 1, 0, 0),
        (-sin(angulo_y), 0, cos(angulo_y), 0),
        (0, 0, 0, 1)
    ])

    matriz_rotacao_z = np.array([
        (cos(angulo_z), -sin(angulo_z), 0, 0),
        (sin(angulo_z), cos(angulo_z), 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1)
    ])

    matriz_rotacao = matriz_rotacao_x @ matriz_rotacao_y @ matriz_rotacao_z

    plano_projetado = []
    for objetos in universo:
        for aresta in objetos:
            vertice1, vertice2 = aresta
            vertice_t1 = np.append(vertice1, 1) @ matriz_rotacao.T
            vertice_t2 = np.append(vertice2, 1) @ matriz_rotacao.T
            vertice_p1 = projetar_ponto(vertice_t1, distancia)
            vertice_p2 = projetar_ponto(vertice_t2, distancia)
            nova_aresta = np.array([vertice_p1, vertice_p2])
            plano_projetado.append(nova_aresta)

    # Desenhar
    tela.fill( (0, 0, 0) )
    for aresta in plano_projetado:
        # print("Aresta: ", aresta)
        vertice1, vertice2 = aresta
        # print(vertice1, vertice2)
        pygame.draw.line(tela, YELLOW, vertice1, vertice2, 2)

    pygame.display.update()

    # Capturar eventos
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            executando = False
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_UP:
                incremento_distancia = 1
            elif e.key == pygame.K_DOWN:
                incremento_distancia = -1
            elif e.key == pygame.K_a:
                incremento_angulo_x = -0.001
            elif e.key == pygame.K_d:
                incremento_angulo_x = 0.001
            elif e.key == pygame.K_w:
                incremento_angulo_y = -0.001
            elif e.key == pygame.K_s:
                incremento_angulo_y = 0.001
            elif e.key == pygame.K_q:
                incremento_angulo_z = -0.001
            elif e.key == pygame.K_e:
                incremento_angulo_z = 0.001                
        elif e.type == pygame.KEYUP:
            if e.key in [pygame.K_UP, pygame.K_DOWN]:
                incremento_distancia = 0
            if e.key in [pygame.K_a, pygame.K_d]:
                incremento_angulo_x = 0
            if e.key in [pygame.K_w, pygame.K_s]:
                incremento_angulo_y = 0
            if e.key in [pygame.K_q, pygame.K_e]:
                incremento_angulo_z = 0
