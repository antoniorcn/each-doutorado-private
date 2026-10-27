from typing import Tuple
import pygame
from ImageEngineAula08 import ImageEngine

tela = pygame.display.set_mode( (800, 600), 0, 32)

def localizar_tipo_linha( imagem : pygame.Surface, pos : Tuple[int, int]):
    engine = ImageEngine( imagem )
    w, h = imagem.get_size()
    cor_procurada = 0
    linha_encontrada = False
    x1, y1 = pos
    for y in range(y1, h):
        if linha_encontrada:
            break
        for x in range(x1, w):
            if linha_encontrada:
                break
            arr1 = engine.verificar_vizinhanca(imagem, (x, y), 1)
            if arr1[0, 1] == cor_procurada and arr1[2, 1] == cor_procurada:
                print(f"Localizada linha vertical | no pixel ({x}, {y})")
                linha_encontrada = True
            elif arr1[1, 0] == cor_procurada and arr1[1, 2] == cor_procurada:
                print(f"Localizada linha horizontal - no pixel ({x}, {y})")
                linha_encontrada = True
            elif arr1[0, 0] == cor_procurada and arr1[2, 2] == cor_procurada:
                print(f"Localizada linha diagonal esquerda para direita \\ no pixel ({x}, {y})")
                linha_encontrada = True
            elif arr1[0, 2] == cor_procurada and arr1[2, 0] == cor_procurada:
                print(f"Localizada linha diagonal direita para esquerda / no pixel ({x}, {y})")
                linha_encontrada = True


tela_branca_com_linhas = pygame.image.load("./assets/tela_branca_com_linhas.png").convert_alpha()
executando = True
image_pos = (50, 200)
while executando:

    tela.fill( (0, 0, 0) )
    tela.blit(tela_branca_com_linhas, image_pos)
    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            executando = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            click_pos = (event.pos[0] - image_pos[0], event.pos[1] - image_pos[1])
            localizar_tipo_linha(tela_branca_com_linhas, click_pos)
