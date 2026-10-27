import pygame
from random import randint



pygame.init()


tela = pygame.display.set_mode((800, 600), 0 ,32)

def funcao_linha( x, y, d_x, d_y, b):
    return d_y * x - d_x * y + b * d_x

def swap_if_need( p1 : int, p2 : int ):
    if p1 > p2:
        temp = p1 
        p1 = p2
        p2 = temp
    return (p1, p2)

def desenhar_linha(x1 : int, y1 : int, x2 : int, y2 : int):
    x1, x2 = swap_if_need(x1, x2)
    y1, y2 = swap_if_need(y1, y2)
    x_p = x1
    y_p = y1
    pontos = [(x_p, y_p)]
    d_x = abs(x1 - x2)
    d_y = abs(y1 - y2)
    b = -d_y * x_p + y_p * d_x
    p = 2 * d_y - d_x
    for _ in range(d_x):
        pm_x = x_p + 1
        pm_y = y_p + 0.5
        # valor = funcao_linha(pm_x, pm_y, d_x, d_y, b)
        # print(p)
        if p < 0:
            p = p + 2 * d_y
            x_p = x_p + 1
        else:
            p = p + 2 * d_y - 2 * d_x
            x_p = x_p + 1
            y_p = y_p + 1
        pontos.append((x_p, y_p))
    return pontos


def desenhar_circunferencia_octante( c_x : int, c_y : int, r : int ):
    x = 0
    y = r
    d = x ** 2 +  y ** 2 - r ** 2
    pontos = []
    while (x) < y:
        pontos.append( (c_x + x, c_y + y) )
        pontos.append( (c_x - x, c_y + y) )
        pontos.append( (c_x + x, c_y - y) )
        pontos.append( (c_x - x, c_y - y) )
        pontos.append( (c_x - y, c_y + x) )
        pontos.append( (c_x + y, c_y + x) )
        pontos.append( (c_x + y, c_y - x) )
        pontos.append( (c_x - y, c_y - x) )
        # print(d)
        x = x + 1
        if d < 0:
            d = d + 4 * x + 6
        else:
            d = d + 4 * (x - y) + 10
            y = y - 1
    return pontos


executando = True
draw_pontos = []
draw_pontos.extend(desenhar_linha(150, 400, 600, 300))
draw_pontos.extend(desenhar_circunferencia_octante(400, 300, 200))

colors = []
for p in draw_pontos:
    colors.append( (255, min(p[0] / 2, 255), min(p[1] / 2, 255)) )

while executando:

    # calcular regras

    # Desenhar pontos
    for idx, p in enumerate(draw_pontos):
        # print(colors[idx])
        tela.set_at(p, colors[idx])

    pygame.display.update()

    # Capturar eventos

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            executando = False