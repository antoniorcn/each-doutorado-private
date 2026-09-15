import pygame

WIDTH = 800
HEIGHT = 600


tela = pygame.display.set_mode( (WIDTH, HEIGHT), 0 , 32 )
executando = True

display_file = []

while executando:

    # Calcular Regras

    # Desenhar Tela



    # Capturar Eventos
    for e in pygame.event.get(): 
        if e.type == pygame.QUIT:
            executando = False
