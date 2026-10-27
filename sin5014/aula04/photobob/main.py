from typing import List, Tuple
from collections.abc import Callable
import pygame
from ImageEngine import ImageEngine


tela = pygame.display.set_mode( (800, 600), 0, 32)

def somar_pixel_rgb( pixel1 : List, pixel2: int ) -> List:
    p1_r, p1_g, p1_b = pixel1
    p2_r = pixel2 & 255
    p2_g = (pixel2 << 8) & 255
    p2_b = (pixel2 << 16) & 255
    s_r = min((p1_r + p2_r), 255)
    s_g = min((p1_g + p2_g), 255)
    s_b = min((p1_b + p2_b), 255)
    # print(f"P1({p1_r}, {p1_g}, {p1_b})  P2({p2_r}, {p2_g}, {p2_b})   Soma({s_r}, {s_g}, {s_b})")
    return [s_r, s_g, s_b]


def somar_pixel_color_trunc( pixel1 : List, pixel2: pygame.Color ) -> List:
    p1_r, p1_g, p1_b = pixel1
    p2_r = pixel2.r
    p2_g = pixel2.g
    p2_b = pixel2.b
    s_r = min((p1_r + p2_r), 255)
    s_g = min((p1_g + p2_g), 255)
    s_b = min((p1_b + p2_b), 255)
    # print(f"P1({p1_r}, {p1_g}, {p1_b})  P2({p2_r}, {p2_g}, {p2_b})   Soma({s_r}, {s_g}, {s_b})")
    return [s_r, s_g, s_b]


def somar_pixel_color( pixel1 : List, pixel2: pygame.Color ) -> List:
    p1_r, p1_g, p1_b = pixel1
    p2_r = pixel2.r
    p2_g = pixel2.g
    p2_b = pixel2.b
    s_r = p1_r + p2_r
    s_g = p1_g + p2_g
    s_b = p1_b + p2_b
    # print(f"P1({p1_r}, {p1_g}, {p1_b})  P2({p2_r}, {p2_g}, {p2_b})   Soma({s_r}, {s_g}, {s_b})")
    return [s_r, s_g, s_b]


def consolidador_soma( _ : pygame.Surface, raio : int, totalizador : List ) -> List:
    return totalizador


def consolidador_media( _ : pygame.Surface, raio : int, totalizador : List ) -> List:
    number_pixels = (2 * ((raio ** 2) + 1)) + (2 * (raio ** 2))
    novo_totalizador = [min(item / number_pixels, 255) for item in totalizador]
    return novo_totalizador

def consolidador_mediana( _ : pygame.Surface, __ : int, totalizador : List[List] ) -> pygame.Color:
    reds = []
    greens = []
    blues = []
    for l in totalizador:
        for pixel_color in l:
            reds.append( pixel_color.r)
            greens.append( pixel_color.g)
            blues.append( pixel_color.b)
    reds.sort()
    greens.sort()
    blues.sort()
    r = reds[len(reds) // 2]
    g = greens[len(greens) // 2]
    b = blues[len(blues) // 2]
    return pygame.Color((r, g, b))


def recalcular_vizinhos(imagem : pygame.Surface,\
                        pos : Tuple[int, int], raio : int,
                        algoritmo : Callable[[List, pygame.Color], List],
                        consolidador: Callable[[pygame.Surface, int, List], List]  ) -> List:
    # pixels = pygame.PixelArray( imagem )
    p_x, p_y = pos
    totalizador = [0, 0, 0]
    largura, altura = imagem.get_size()
    for lin in range(p_y - raio, p_y + raio + 1):
        for col in range(p_x - raio, p_x + raio + 1):
            # print(soma, pixels[col][lin])
            if lin > 0 and lin < altura and col > 0 and col < largura:
                # print("Color: ", imagem.get_at((col, lin)))
                totalizador = algoritmo(totalizador, imagem.get_at((col, lin)))
    # del pixels
    return consolidador(imagem, raio, totalizador)


def vizinhos(imagem : pygame.Surface,
             pos : Tuple[int, int], raio : int,
             consolidador: Callable[[pygame.Surface, int, List[List]], pygame.Color]  ) -> pygame.Color:
    # pixels = pygame.PixelArray( imagem )
    p_x, p_y = pos
    lista_vizinhos = []
    largura, altura = imagem.get_size()
    for lin in range(p_y - raio, p_y + raio + 1):
        coluna = []
        for col in range(p_x - raio, p_x + raio + 1):
            # print(soma, pixels[col][lin])
            if lin > 0 and lin < altura and col > 0 and col < largura:
                coluna.append( imagem.get_at((col, lin)) )
        lista_vizinhos.append(coluna)
                
    # del pixels
    return consolidador(imagem, raio, lista_vizinhos)


papagaio = pygame.image.load("./assets/papagaio-rosa.png").convert_alpha()
tela_branca_com_linhas = pygame.image.load("./assets/tela_branca_com_linhas.png").convert_alpha()

papagaio_redim = pygame.transform.scale(tela_branca_com_linhas, (200, 200))

novo_papagaio = pygame.Surface((200, 200))


for y in range(0, 200):
    for x in range(0, 200):
        # nova_cor = recalcular_vizinhos(papagaio_redim, (x, y), 1,
        # somar_pixel_color, consolidador_media)
        # print(soma)
        nova_cor = vizinhos(papagaio_redim, (x, y), 1, consolidador_mediana)
        # print("Nova Cor: ", nova_cor)
        novo_papagaio.set_at( (x, y), nova_cor)

# hist = Histogramas(papagaio_redim)
# hist.gerar_histograma()
# papagaio_equalizado = hist.equalizar()

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
# print("Arr1: ", arr1)
executando = True
image_pos = (50, 200)
while executando:

    tela.fill( (0, 0, 0) )
    tela.blit(papagaio_redim, image_pos)
    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            executando = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            click_pos = (event.pos[0] - image_pos[0], event.pos[1] - image_pos[1])
            localizar_tipo_linha(tela_branca_com_linhas, click_pos)
