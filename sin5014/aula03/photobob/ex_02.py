import pygame
from ImageEngine import ImageEngine


tela = pygame.display.set_mode( (1024, 600), 0, 32)

papagaio = pygame.image.load("./assets/papagaio-rosa.png").convert_alpha()

papagaio_redim = pygame.transform.scale(papagaio, (200, 200))

img_eng = ImageEngine(papagaio_redim)
img_eng.gerar_histograma()
papagaio_equalizado = img_eng.equalizar()
hist_img = img_eng.hist_draw()
papagaio_claro = img_eng.mudar_brilho( -50 )

executando = True

while executando:

    tela.fill( (0, 0, 0) )
    tela.blit(papagaio_redim, (50, 50))
    tela.blit(hist_img, (350, 50))
    tela.blit(papagaio_equalizado, (650, 50))
    tela.blit(papagaio_claro, (50, 350))
    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            executando = False



