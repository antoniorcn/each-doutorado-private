import pygame
import numpy as np
import math
import random

# Inicializar o pygame
pygame.init()


# Parâmetros
escala_erro = 1.0
offset_x, offset_y = 0, 0
angle_x, angle_y = math.radians(30), math.radians(45)
mouse_drag_mode = None  # "rotate" ou "pan"
last_mouse_pos = None

# Histórico de erro
dados_historico = {}


# Configurações da janela
width, height = 1280, 700
tela = pygame.display.set_mode((width, height))
tela_top_bar = pygame.Surface(size=(width, 200), flags=0, depth=32)
legenda_height = height - tela_top_bar.get_height()
tela_legenda = pygame.Surface(size=(200, legenda_height), flags=0, depth=32)
chart_width = width - tela_legenda.get_width()
chart_height = legenda_height
tela_grafico = pygame.Surface(size=(chart_width, chart_height), flags=0, depth=32)
tela_malha = pygame.Surface(size=(chart_width, chart_height), flags=0, depth=32)

pygame.display.set_caption("Perceptron 2D - Visualização em Pygame")
fonte_normal = pygame.font.Font("arial.ttf", 18)
fonte_grande = pygame.font.Font("arial.ttf", 32)

# Cores
WHITE = (255, 255, 255)
RED   = (255, 0, 0)
BLUE  = (0, 0, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
BISQUE = (255, 228, 196)
NAVY = (0, 0, 128)
DIMGRAY = (105, 105, 105)
ALMOSTBLACK = (40, 40, 40)

def erro(x, w):
    return math.sin(x) * math.cos(w) * escala_erro

def rotacionar_ponto(x, y, z, ax, ay):
    cos_y, sin_y = math.cos(ay), math.sin(ay)
    xz = x * cos_y + z * sin_y
    zz = -x * sin_y + z * cos_y
    cos_x, sin_x = math.cos(ax), math.sin(ax)
    yz = y * cos_x - zz * sin_x
    zz = y * sin_x + zz * cos_x
    return xz, yz, zz

def project_point(x, y, z):
    scale = 20
    x, y, z = rotacionar_ponto(x, y, z, angle_x, angle_y)
    px = int(chart_width / 2 + x * scale + offset_x)
    py = int(chart_height / 2 + y * scale + offset_y)
    return px, py

def gerar_malha():
    # print("Dados Historico: ")
    # for i in dados_historico.items():
    #     print(i)
    malha = []

    # # Descobre o menor e maior valor de W e X1
    # w1_menor = None
    # w2_menor = None
    # w2_maior = None
    # w1_maior = None
    # for k_indice, k in enumerate(dados_historico.keys()):
    #     v_dados = dados_historico[k]
    #     if k_indice == 0:
    #         w1_menor = v_dados[0]
    #         w1_maior = v_dados[0]
    #         w2_menor = v_dados[1]
    #         w2_maior = v_dados[1]
    #     if v_dados[0] < w1_menor:
    #         w1_menor = v_dados[0]
    #     if v_dados[0] > w1_maior:
    #         w1_maior = v_dados[0]
    #     if v_dados[1] < w2_menor:
    #         w2_menor = v_dados[1]
    #     if v_dados[1] > w2_maior:
    #         w2_maior = v_dados[1]
    # print(f"W1_menor: {w1_menor}\tW1_maior: {w1_maior}\tW2_menor: {w2_menor}\tW2_maior: {w2_maior}")
    
    # w1_range = range(int(w1_menor) - 3, int(w1_maior) + 3)
    # # w_range = range(int(w1_menor), int(w1_maior))
    # w2_range = range(int(w2_menor) - 3, int(w2_maior) + 3)
    w1_range = sorted( set(k[0] for k in dados_historico.keys()))
    w2_range = sorted( set(k[1] for k in dados_historico.keys()))
    #  np.zeros( (len(w1_range), len(w2_range)) )
    matriz = [[[] for w2 in w2_range] for w1 in w1_range ]
    print("Matriz: ", matriz)
    for (w1, w2), valor in dados_historico.items():
        linha = w1_range.index(w1)
        coluna = w2_range.index(w2)
        print(f"Dados extraidos Linha: {linha}  Coluna: {coluna}   W1: {w1}   W2: {w2}   Valor: {valor}")
        matriz[linha][coluna] = (w1, w2, valor)
    print("Matriz: ", matriz)
    if len(matriz) > 1:
        for linha_item in matriz:
            linha = []
            for coluna_item in linha_item:
                # z = erro(local_x, local_w)
                local_w1, local_w2, local_erro = coluna_item
                px, py = project_point(local_w1, local_w2, local_erro)
                linha.append((px, py))
            malha.append(linha)
    return malha


# Escala do plano cartesiano para tela
def to_screen_coords(screen, x, y):
    return int(x / 10 * screen.get_width()), int(screen.get_height() - (y / 10 * screen.get_height()))

# Gerar dados
def gerar_dados(n=50, seed=42):
    np.random.seed(seed)
    X = []
    y = []
    for _ in range(n):
        x1 = np.random.uniform(0, 10)
        x2 = np.random.uniform(0, 10)
        label = 1 if x2 > x1 else 0
        X.append([x1, x2])
        y.append(label)
    return np.array(X), np.array(y)

instancias = 50
X, y = gerar_dados(n = instancias)
X_b = np.hstack((X, np.ones((X.shape[0], 1))))  # adicionar bias
w = np.random.random(X_b.shape[1])  # pesos: w1, w2, bias
dados = list(zip(X_b, y))
erros = np.zeros(X.shape[0])
dados_plus_erros = list(zip(X_b, y, erros))
taxa = 0.001
epocas = 20
current_epoca = 0

# Função de ativação
def step(x):
    return 1 if x >= 0 else 0

# Treinar uma época e retornar a reta
def treinar_uma_epoca(w, X_b, y):
    for xi, yi in zip(X_b, y):
        z = np.dot(w, xi)
        y_pred = step(z)
        erro = yi - y_pred
        w += taxa * erro * xi
    return w

def treinar(i, w, X_b, y):
    xi, yi = dados[i]
    z = np.dot(w, xi)
    y_pred = step(z)
    error = yi - y_pred
    dados_historico[(w[0], w[1])] = error
    w += taxa * error * xi
    return w, error

# Desenhar tudo
def desenhar_tela(screen, w):
    screen.fill(ALMOSTBLACK)

    # Desenhar pontos
    for i in range(len(X)):
        color = BLUE if y[i] == 1 else RED
        x_screen, y_screen = to_screen_coords(screen, X[i][0], X[i][1])
        pygame.draw.circle(screen, color, (x_screen, y_screen), 5)

    # Desenhar reta
    if w[1] != 0:
        x1, x2 = 0, 10
        y1 = -(w[0] * x1 + w[2]) / w[1]
        y2 = -(w[0] * x2 + w[2]) / w[1]
        start = to_screen_coords(screen, x1, y1)
        end = to_screen_coords(screen, x2, y2)
        pygame.draw.line(screen, YELLOW, start, end, 2)

def desenhar_legenda(screen, concluido=False):
    fonte_grande_espaco = fonte_grande.get_height()
    fonte_normal_espaco = fonte_normal.get_height()
    pos_y = 0
    screen.fill(DIMGRAY)
    if concluido:
        screen.blit( fonte_grande.render("Treinamento", True, YELLOW), (10, pos_y))
        pos_y += fonte_grande_espaco
        screen.blit( fonte_grande.render("Concluido", True, YELLOW), (30, pos_y))
        pos_y += fonte_grande_espaco
    else:
        screen.blit( fonte_grande.render("Treinando", True, YELLOW), (30, pos_y))
        pos_y += fonte_grande_espaco
    screen.blit( fonte_normal.render(f"Epoca: {current_epoca}/{epocas}", True, YELLOW), (0, pos_y))
    pos_y += fonte_normal_espaco
    screen.blit( fonte_normal.render(f"Instancia: {indice_instancia}/{instancias}", True, YELLOW), (0, pos_y))
    pos_y += fonte_normal_espaco
    screen.blit( fonte_normal.render(f"Peso (1): {w[0]:5.2f}", True, YELLOW), (0, pos_y))
    pos_y += fonte_normal_espaco
    screen.blit( fonte_normal.render(f"Peso (2): {w[1]:5.2f}", True, YELLOW), (0, pos_y))
    pos_y += fonte_normal_espaco
    screen.blit( fonte_normal.render(f"Peso (B): {w[2]:5.2f}", True, YELLOW), (0, pos_y))
    pos_y += fonte_normal_espaco

def desenhar_malha(screen):
    # Desenhar malha
    # print(malha)
    if len(malha) > 0 and len(malha[0]) > 0:
        screen.fill(ALMOSTBLACK)
        for linha in malha:
            pygame.draw.lines(screen, GREEN, False, linha, 1)
        for i in range(len(malha[0])):
            coluna = [linha[i] for linha in malha]
            pygame.draw.lines(screen, GREEN, False, coluna, 1)

# Loop principal
clock = pygame.time.Clock()
print("Clock: ", clock)
running = True
update_in_miliseconds = 1
ticks = pygame.time.get_ticks()
miliseconds_passed = ticks
seconds = 0
indice_instancia = 0
concluido = False
view_mode = "2d"
while running:
    # Calcular regras
    ticks = pygame.time.get_ticks()
    miliseconds_passed = ticks
    if (miliseconds_passed - seconds) > update_in_miliseconds and not concluido: 
        seconds = miliseconds_passed

        if current_epoca <= epocas:
            w, err = treinar(indice_instancia, w, X_b, y)
            indice_instancia += 1
            if indice_instancia >= instancias:
                current_epoca += 1
                indice_instancia = 0
        else:
            concluido = True
            current_epoca = epocas
            indice_instancia = 0

    malha = gerar_malha()

    # Limpar a tela
    tela.fill(BLACK)
    desenhar_legenda(tela_legenda, concluido)
    desenhar_tela(tela_grafico, w)
    desenhar_malha(tela_malha)
    
    if view_mode == "2d": 
        tela.blit(tela_grafico, (0, tela_top_bar.get_height()))
    elif view_mode == "3d":
        tela.blit(tela_malha, (0, tela_top_bar.get_height()))
    tela.blit(tela_legenda, (chart_width, tela_top_bar.get_height()))
    pygame.display.update()

    # Capturar eventos
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                escala_erro += 0.1
            elif event.key == pygame.K_DOWN:
                escala_erro -= 0.1
            elif event.key == pygame.K_3:
                view_mode = "3d"
            elif event.key == pygame.K_2:
                view_mode = "2d"
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Botão esquerdo: rotação
                mouse_drag_mode = "rotate"
                last_mouse_pos = pygame.mouse.get_pos()
            elif event.button == 3:  # Botão direito: pan
                mouse_drag_mode = "pan"
                last_mouse_pos = pygame.mouse.get_pos()

        elif event.type == pygame.MOUSEBUTTONUP:
            mouse_drag_mode = None

        elif event.type == pygame.MOUSEMOTION and mouse_drag_mode:
            mx, my = pygame.mouse.get_pos()
            lx, ly = last_mouse_pos
            dx, dy = mx - lx, my - ly

            if mouse_drag_mode == "rotate":
                angle_y += dx * 0.01  # rotação horizontal
                angle_x += dy * 0.01  # rotação vertical

            elif mouse_drag_mode == "pan":
                offset_x += dx
                offset_y += dy

            last_mouse_pos = (mx, my)

pygame.quit()
