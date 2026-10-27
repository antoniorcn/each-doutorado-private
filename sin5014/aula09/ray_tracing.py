from typing import List, Tuple

import pygame
from random import randint

pygame.init()
tela = pygame.display.set_mode((800, 600), 0 ,32)


def esfera( x : float, y : float, z : float, r : float, cor : Tuple )->\
    Tuple[float, float, float, float, Tuple]:
    return (x, y, z, r, cor)


def colidiu_ponto_esfera(ponto_a : Tuple[float, float, float],
                         esfera_a : Tuple[float, float, float, float, Tuple] ):
    x, y, z = ponto_a
    ex, ey, ez, er, _ = esfera_a
    return (x - ex) ** 2 + (y - ey) ** 2 + (z - ez) ** 2 < er ** 2

def colidiu_esfera_esfera(esfera_a : Tuple[float, float, float, float, Tuple],
                          esfera_b : Tuple[float, float, float, float, Tuple]):
    e_ax, e_ay, e_az, e_ar, _ = esfera_a
    e_bx, e_by, e_bz, e_br, _ = esfera_b
    return (e_ax - e_bx) ** 2 + (e_ay - e_by) ** 2 + (e_az - e_bz) ** 2 < (e_ar + e_br)



def ray_tracing(screen : pygame.Surface, mundo : List, observador : Tuple[float, float, float],
                direcao : Tuple[float, float, float],
                min : float, max : float, tamanho_janela : float) -> List[Tuple[float, float, Tuple]]:
    w, h = screen.get_size()
    window = [-tamanho_janela, -tamanho_janela, tamanho_janela, tamanho_janela]
    window_w = tamanho_janela * 2
    window_h = tamanho_janela * 2
    raio_inicial = 0.1
    raio_minimo = 0.01
    z_step = 0.1
    observador_x, observador_y, observador_z = observador
    pontos_2d = []
    precisao = 100
    
    precisao_w = int(window_w) * precisao
    precisao_h = int(window_h) * precisao
    print(f"Precisao: {(precisao_w, precisao_h)}")
    x = window[0]
    for _ in range(precisao_w):
        y = window[1]
        for _ in range(precisao_h):
            mx = (x - observador_x) / (min - observador_z)
            my = (y - observador_y) / (min - observador_z)
            z = min
            # print(f"Ponto: {(x, y, z)}")
            colision = False
            while z < max and not colision:
                ponto_ray = (x + mx * (z - observador_z), y + my * (z - observador_z), z)
                for elemento in mundo:
                    colisao = colidiu_ponto_esfera(ponto_ray, elemento)
                    if colisao:
                        _, _, _, _, cor = elemento
                        nx = ((x + tamanho_janela) / window_w) * w
                        ny = ((y + tamanho_janela) / window_h) * h
                        np = (nx, ny, cor)
                        # print(f"Colidiu: {np}\tPonto: {(x, y)}")
                        pontos_2d.append( np )
                        break
                z = z + z_step
            y = y + (1.0 / precisao)
        x = x + (1.0 / precisao)
    return pontos_2d

def rescale_pontos( screen : pygame.Surface, tamanho_janela : float, lista_pontos : List[Tuple] ) -> List[Tuple]:
    w, h = screen.get_size()
    window_w = tamanho_janela * 2
    window_h = tamanho_janela * 2
    novos_pontos = []
    for p in lista_pontos:
        wx, wy, wcor = p
        nx = ((wx + tamanho_janela) / window_w) * w
        ny = ((wy + tamanho_janela) / window_h) * h
        np = (nx, ny, wcor)
        print(np)
        novos_pontos.append( np )
    return novos_pontos


world = [ esfera(0.0, 0.0, 2.0, 5.0, (255, 255, 0)) ]

executando = True
screen_pontos = ray_tracing(tela, mundo=world, observador=(0, 0, 0), direcao=(0, 0, 10),
                            min=1.0, max=10.0, tamanho_janela=5)
# screen_pontos = rescale_pontos( tela, 5, pontos)
while executando:

    # calcular regras

    # Desenhar na tela
    for ponto in screen_pontos:
        px, py, cor = ponto
        tela.set_at((int(px), int(py)), cor)
    pygame.display.update()

    # Capturar eventos

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            executando = False