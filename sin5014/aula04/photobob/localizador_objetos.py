from typing import List, Tuple
from collections.abc import Callable
import sys
import pygame
from ImageEngineAula08 import ImageEngine

# Aumenta o tamanho da memoria Stack para permitir o uso do algoritmo
print(sys.getrecursionlimit())
sys.setrecursionlimit(10000)
print(sys.getrecursionlimit())


tela = pygame.display.set_mode( (800, 600), 0, 32)

imagem = pygame.image.load("./assets/3objetos.bmp").convert_alpha()

executando = True
image_pos = (50, 200)

while executando:

    tela.fill( (0, 0, 0) )
    tela.blit(imagem, image_pos)
    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            executando = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            print("Procurando objetos...")
            print(ImageEngine.identificar_objetos(imagem))
