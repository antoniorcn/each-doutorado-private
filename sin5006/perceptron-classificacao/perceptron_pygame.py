import pygame
import numpy as np

# Inicializar o pygame
pygame.init()

# Configurações da janela
width, height = 1280, 700
tela = pygame.display.set_mode((width, height))
tela_top_bar = pygame.Surface(size=(width, 200), flags=0, depth=32)
legenda_height = height - tela_top_bar.get_height()
tela_legenda = pygame.Surface(size=(200, legenda_height), flags=0, depth=32)
chart_width = width - tela_legenda.get_width()
chart_height = legenda_height
tela_grafico = pygame.Surface(size=(width, chart_height), flags=0, depth=32)
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
print("Dados: ")
for i in dados:
    print(i)
taxa = 0.001
epocas = 200
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
    erro = yi - y_pred
    w += taxa * erro * xi
    return w

def desenhar_titulo(screen):
    screen.fill(BLACK)
    fonte_grande_espaco = fonte_grande.get_height()
    pos_y = 0
    screen.blit( fonte_grande.render("Exemplo de Treinamento", True, YELLOW), (10, pos_y))
    pos_y += fonte_grande_espaco
    screen.blit( fonte_grande.render("2 classes com instâncias aleatórias", True, YELLOW), (30, pos_y))

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
while running:
    # Calcular regras
    ticks = pygame.time.get_ticks()
    miliseconds_passed = ticks
    if (miliseconds_passed - seconds) > update_in_miliseconds and not concluido: 
        seconds = miliseconds_passed

        if current_epoca <= epocas:
            w = treinar(indice_instancia, w, X_b, y)
            indice_instancia += 1
            if indice_instancia >= instancias:
                current_epoca += 1
                indice_instancia = 0
        else:
            concluido = True
            current_epoca = epocas
            indice_instancia = 0

    # Desenhar a tela
    tela.fill(BLACK)
    desenhar_tela(tela_grafico, w)
    desenhar_legenda(tela_legenda, concluido)
    tela.blit(tela_grafico, (0, 160))
    tela.blit(tela_legenda, (600, 160))
    pygame.display.update()

    # Capturar eventos
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False



pygame.quit()
