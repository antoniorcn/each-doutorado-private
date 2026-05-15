from typing import Tuple

COLUNAS = 8
LINHAS = 8

tabuleiro = [[0 for _ in range(COLUNAS)] for _ in range(LINHAS)] 

MAXIMO_MOVIMENTOS = COLUNAS * LINHAS

movimentos_possiveis = [
    (-1, 2), (-2, 1), (-2, -1), (-1, -2),
    (1, -2), (2, -1), (2, 1),  (1, 2)
]

def nova_posicao( y : int, x : int,
                 movimento : Tuple[int, int]):
    nx : int = x + movimento[1]
    ny : int = y + movimento[0]
    return (ny, nx)

def tentar( y : int, x : int ) -> bool:
    numero_movimento = tabuleiro[y][x]
    print(f"Tentando movimento ({numero_movimento}) [{y}][{x}]")
    if numero_movimento == MAXIMO_MOVIMENTOS - 1:
        return True
    for movimento in movimentos_possiveis:
        ny, nx = nova_posicao(y, x, movimento)
        if 0 <= nx < 8 and 0 <= ny < 8\
            and tabuleiro[ny][nx] == 0:
            tabuleiro[ny][nx] = tabuleiro[y][x] + 1
            if tentar(ny, nx):
                return True
            tabuleiro[ny][nx] = 0
    return False

tentar(0, 0)

for linha in range(LINHAS):
    for coluna in range(COLUNAS):
        print(f"| {tabuleiro[linha][coluna]:3} " , end="")
    print("-" * (COLUNAS * 6))