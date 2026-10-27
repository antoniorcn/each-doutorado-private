import math
import pygame

pygame.init()


screen = pygame.display.set_mode( (800, 600), 0, 32)

def f1(x : float, y: float) -> float:
    return x*x+y*y-4.0

def f2(x : float, y: float) -> float:
    return x*x-y*y-4.0

def f3(x : float, y: float) -> float:
    return x*x+math.sin(x*y)-2.0


def gerar_curva(tela : pygame.Surface, x_inicial : float, y_inicial : float,  raio = 2.0, step = 0.1):
    t = 0

    x = raio * math.cos( t )
    y = raio * math.sin( t )
    rodando = True
    while t <= 2 * math.pi and rodando:

        t = t + step
        if t >= 2 * math.pi:
            t = 2 * math.pi
            rodando = False

        x_next = raio * math.cos( t )
        y_next = raio * math.sin( t )

        pygame.draw.line(tela, (255, 255, 0), (x_inicial + x, y_inicial + y),
                         (x_inicial + x_next, y_inicial + y_next), 2)
        pygame.draw.circle(tela, (255, 0, 0), (x_inicial + x, y_inicial + y), 2, 0)

        x = x_next
        y = y_next


def desenhar_grade(tela, grade : )

objetos = []

executando = True

while executando:

    # Desenhar na tela
    screen.fill( (0, 0, 0))
    gerar_curva(screen, 300, 250, 200, step=0.4 )
    pygame.display.update()

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            executando = False
