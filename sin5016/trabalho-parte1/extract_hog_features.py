"""
Script para extrair features HOG (Histogram of Oriented Gradients) de imagens PNG e JPG.
Suporta wildcards para seleção de múltiplos arquivos.
"""

import argparse
import glob
import os
import sys
import csv
from typing import List, Optional, Tuple, Union
import numpy as np
import cv2
import pywt

FEATURE_COLUMN_PREFIX = 'in_'

def format_value_for_csv(value: Union[str, float], decimal_sep: str) -> str:
    """Converte valores numéricos para string usando separador decimal customizado."""
    if isinstance(value, (float, np.floating)):
        text = f"{float(value)}"
        if decimal_sep != '.':
            text = text.replace('.', decimal_sep)
        return text
    return str(value)

def apply_wavelet_decomposition(
    image: np.ndarray,
    wavelet: str,
    levels: int
) -> np.ndarray:
    """
    Aplica decomposição wavelet múltiplas vezes e retorna apenas a sub-banda LL.
    
    Args:
        image: Imagem em escala de cinza (numpy array 2D)
        wavelet: Tipo de wavelet a ser usado (ex: 'haar', 'db4', 'bior2.2')
        levels: Número de vezes que o wavelet será aplicado
        
    Returns:
        Imagem reduzida contendo apenas a sub-banda LL após N níveis de decomposição
    """
    if levels <= 0:
        return image

    # Converte para float64 para processamento wavelet
    current_image: np.ndarray = image.astype(np.float64)

    for level in range(levels):
        # Verifica se a imagem ainda tem tamanho suficiente para decomposição
        if current_image.shape[0] < 2 or current_image.shape[1] < 2:
            print(f"Aviso: Imagem muito pequena para aplicar mais decomposições. "
                  f"Aplicados {level} nível(is) ao invés de {levels}.", file=sys.stderr)
            break

        # Aplica decomposição wavelet 2D
        # Retorna: (cA, (cH, cV, cD)) onde:
        #   cA = coeficientes de aproximação (LL)
        #   cH = coeficientes horizontais (LH)
        #   cV = coeficientes verticais (HL)
        #   cD = coeficientes diagonais (HH)
        coeffs: Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]] = pywt.dwt2(
            current_image,
            wavelet,
            mode='symmetric'
        )
        # Extrai apenas a sub-banda LL (coeficientes de aproximação)
        cA: np.ndarray = coeffs[0]
        current_image = cA
    return current_image


def extract_hog_features(
    image_path: str,
    hog_descriptor: cv2.HOGDescriptor,
    wavelet: Optional[str] = None,
    wavelet_levels: int = 0
) -> Optional[np.ndarray]:
    """
    Extrai features HOG de uma imagem, opcionalmente aplicando wavelet antes.
    
    Args:
        image_path: Caminho para a imagem
        hog_descriptor: Descritor HOG do OpenCV
        wavelet: Tipo de wavelet a ser usado (None para não usar wavelet)
        wavelet_levels: Número de vezes que o wavelet será aplicado (0 para não usar)
        
    Returns:
        numpy array com as features HOG ou None em caso de erro
    """
    try:
        # Carrega a imagem em escala de cinza
        image: Optional[np.ndarray] = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Erro: Não foi possível carregar a imagem {image_path}", file=sys.stderr)
            return None

        # Aplica wavelet se especificado
        if wavelet is not None and wavelet_levels > 0:
            image = apply_wavelet_decomposition(image, wavelet, wavelet_levels)
            # Converte para uint8 se necessário (pywt retorna float)
            if image.dtype != np.uint8:
                # Normaliza os coeficientes wavelet para o range 0-255
                # Usa min-max normalization
                img_min: float = float(image.min())
                img_max: float = float(image.max())
                if img_max > img_min:
                    image = ((image - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
                else:
                    # Se todos os valores são iguais, define como 128 (meio do range)
                    image = np.full_like(image, 128, dtype=np.uint8)

        # Calcula as features HOG
        features: np.ndarray = hog_descriptor.compute(image)

        # Retorna como array 1D
        return features.flatten()

    except Exception as e:
        print(f"Erro ao processar {image_path}: {str(e)}", file=sys.stderr)
        return None


def get_image_files(path_pattern: str) -> List[str]:
    """
    Obtém lista de arquivos de imagem (PNG e JPG) baseado no padrão fornecido.
    
    Args:
        path_pattern: Caminho com possível wildcard (ex: "imagens/*.jpg")
        
    Returns:
        Lista de caminhos de arquivos de imagem
    """
    # Suporta extensões PNG e JPG (case-insensitive)
    extensions: List[str] = ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']

    # Se o padrão já contém wildcard, usa diretamente
    if '*' in path_pattern or '?' in path_pattern:
        files: List[str] = glob.glob(path_pattern)
    else:
        # Se é um diretório, busca todas as imagens
        if os.path.isdir(path_pattern):
            files = []
            for ext in extensions:
                files.extend(glob.glob(os.path.join(path_pattern, ext)))
        # Se é um arquivo específico, verifica se é imagem
        elif os.path.isfile(path_pattern):
            ext: str = os.path.splitext(path_pattern)[1].lower()
            if ext in ['.png', '.jpg', '.jpeg']:
                files = [path_pattern]
            else:
                files = []
        else:
            # Tenta com wildcards
            files = []
            for ext in extensions:
                pattern: str = path_pattern + ext if not path_pattern.endswith(ext) else path_pattern
                files.extend(glob.glob(pattern))

    # Remove duplicatas e ordena
    files = sorted(list(set(files)))
    return files


def main() -> None:
    """
    Função principal que processa argumentos da linha de comando e extrai features HOG.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description='Extrai features HOG de imagens PNG e JPG usando OpenCV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Exemplos de uso:
            python extract_hog_features.py --input "imagens/*.jpg" --output features.csv
            python extract_hog_features.py --input "fotos/*.png" --output resultado.csv --win-size 128 128 --nbins 18
            python extract_hog_features.py --input "data/" --output features.csv --block-size 32 32
            python extract_hog_features.py -i "*.jpg" -o output.csv --cell-size 16 16
            python extract_hog_features.py --input "*.png" --output features.csv --wavelet-levels 2
            python extract_hog_features.py --input "imagens/*.jpg" --output features.csv --wavelet-levels 3 --wavelet db4
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Caminho para arquivo(s) de imagem. Suporta wildcards (ex: "imagens/*.jpg")'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Nome do arquivo CSV de saída com as features extraídas'
    )

    parser.add_argument(
        '--csv-delimiter',
        type=str,
        default=',',
        help='Delimitador utilizado no CSV de saída (padrão: ,)'
    )

    parser.add_argument(
        '--decimal-sep',
        type=str,
        default='.',
        help='Separador decimal usado ao gravar números (padrão: .)'
    )

    parser.add_argument(
        '--win-size',
        type=int,
        nargs=2,
        default=[64, 64],
        metavar=('WIDTH', 'HEIGHT'),
        help='Tamanho da janela para HOG em pixels (padrão: 64 64). Deve ser múltiplo de cell-size.'
    )

    parser.add_argument(
        '--block-size',
        type=int,
        nargs=2,
        default=[16, 16],
        metavar=('WIDTH', 'HEIGHT'),
        help='Tamanho do bloco para HOG em pixels (padrão: 16 16). Deve ser múltiplo de cell-size.'
    )

    parser.add_argument(
        '--block-stride',
        type=int,
        nargs=2,
        default=[8, 8],
        metavar=('WIDTH', 'HEIGHT'),
        help='Stride do bloco para HOG em pixels (padrão: 8 8). Deve ser múltiplo de cell-size.'
    )

    parser.add_argument(
        '--cell-size',
        type=int,
        nargs=2,
        default=[8, 8],
        metavar=('WIDTH', 'HEIGHT'),
        help='Tamanho da célula para HOG em pixels (padrão: 8 8).'
    )

    parser.add_argument(
        '--nbins',
        type=int,
        default=9,
        help='Número de bins de orientação para HOG (padrão: 9). Valores comuns: 9, 18, 36.'
    )

    parser.add_argument(
        '--wavelet-levels',
        type=int,
        default=0,
        metavar='N',
        help='Número de níveis de decomposição wavelet a aplicar antes do HOG'
        '(padrão: 0, desabilitado). Cada nível reduz a imagem pela metade. '
        'Use valores como 1, 2, 3 para reduzir features.'
    )

    parser.add_argument(
        '--wavelet',
        type=str,
        default='haar',
        help='Tipo de wavelet a ser usado (padrão: haar). '
        'Opções comuns: haar, db4, db8, bior2.2, coif2.'
    )

    args: argparse.Namespace = parser.parse_args()

    # Validação dos parâmetros HOG
    win_size: Tuple[int, int] = tuple(args.win_size)
    block_size: Tuple[int, int] = tuple(args.block_size)
    block_stride: Tuple[int, int] = tuple(args.block_stride)
    cell_size: Tuple[int, int] = tuple(args.cell_size)

    # Validações adicionais
    if len(args.csv_delimiter) != 1:
        print("Erro: --csv-delimiter deve conter exatamente um caractere.", file=sys.stderr)
        sys.exit(1)
    if len(args.decimal_sep) != 1:
        print("Erro: --decimal-sep deve conter exatamente um caractere.", file=sys.stderr)
        sys.exit(1)

    # Valida se os tamanhos são múltiplos apropriados
    if win_size[0] % cell_size[0] != 0 or win_size[1] % cell_size[1] != 0:
        print(f"Erro: win-size ({win_size}) deve ser múltiplo de cell-size ({cell_size})", 
              file=sys.stderr)
        sys.exit(1)

    if block_size[0] % cell_size[0] != 0 or block_size[1] % cell_size[1] != 0:
        print(f"Erro: block-size ({block_size}) deve ser múltiplo de cell-size ({cell_size})", 
              file=sys.stderr)
        sys.exit(1)

    if block_stride[0] % cell_size[0] != 0 or block_stride[1] % cell_size[1] != 0:
        print(f"Erro: block-stride ({block_stride}) deve ser múltiplo de cell-size ({cell_size})", 
              file=sys.stderr)
        sys.exit(1)

    if block_size[0] > win_size[0] or block_size[1] > win_size[1]:
        print(f"Erro: block-size ({block_size}) não pode ser maior que win-size ({win_size})", 
              file=sys.stderr)
        sys.exit(1)

    # Valida parâmetros do wavelet
    if args.wavelet_levels < 0:
        print(f"Erro: wavelet-levels deve ser >= 0 (recebido: {args.wavelet_levels})", 
              file=sys.stderr)
        sys.exit(1)

    # Valida tipo de wavelet
    try:
        if args.wavelet_levels > 0:
            # Testa se o wavelet é válido
            test_array = np.array([[1, 2], [3, 4]], dtype=np.float64)
            pywt.dwt2(test_array, args.wavelet)
    except ValueError as e:
        print(f"Erro: Tipo de wavelet inválido '{args.wavelet}': {str(e)}", file=sys.stderr)
        print("Tipos de wavelet válidos incluem: haar, db1-db20, bior2.2, bior4.4, coif1-coif5", 
              file=sys.stderr)
        sys.exit(1)

    # Obtém lista de arquivos de imagem
    image_files: List[str] = get_image_files(args.input)

    if not image_files:
        print(f"Erro: Nenhuma imagem encontrada com o padrão '{args.input}'", file=sys.stderr)
        sys.exit(1)

    print(f"Encontradas {len(image_files)} imagem(ns) para processar...")
    print(f"Parâmetros HOG: win-size={win_size}, block-size={block_size}, "
        f"block-stride={block_stride}, cell-size={cell_size}, nbins={args.nbins}")
    if args.wavelet_levels > 0:
        print(f"Wavelet: {args.wavelet} com {args.wavelet_levels} nível(is) de decomposição")
    else:
        print("Wavelet: desabilitado")

    # Configura o descritor HOG
    hog: cv2.HOGDescriptor = cv2.HOGDescriptor(
        _winSize=win_size,
        _blockSize=block_size,
        _blockStride=block_stride,
        _cellSize=cell_size,
        _nbins=args.nbins
    )

    # Processa cada imagem
    results: List[List[Union[str, float]]] = []
    feature_size: Optional[int] = None

    # Determina se deve usar wavelet
    wavelet_type: Optional[str] = args.wavelet if args.wavelet_levels > 0 else None

    for image_path in image_files:
        print(f"Processando: {image_path}")
        features: Optional[np.ndarray] = extract_hog_features(
            image_path, 
            hog, 
            wavelet=wavelet_type,
            wavelet_levels=args.wavelet_levels
        )

        if features is not None:
            if feature_size is None:
                feature_size = len(features)
            elif len(features) != feature_size:
                print(f"Aviso: Tamanho de features inconsistente para {image_path}", 
                      file=sys.stderr)
            # Adiciona o caminho do arquivo e as features
            result_row: List[Union[str, float]] = [image_path] + features.tolist()
            results.append(result_row)
        else:
            print(f"Pulando {image_path} devido a erro", file=sys.stderr)

    if not results:
        print("Erro: Nenhuma feature foi extraída com sucesso", file=sys.stderr)
        sys.exit(1)

    # Salva os resultados em CSV
    print(f"Salvando {len(results)} resultado(s) em {args.output}...")

    # Cria cabeçalho do CSV
    header: List[str] = ['image_path'] +\
        [f'{FEATURE_COLUMN_PREFIX}{i}' for i in range(feature_size)]

    with open(args.output, 'w', newline='', encoding='utf-8') as csvfile:
        writer: csv.writer = csv.writer(csvfile, delimiter=args.csv_delimiter)
        writer.writerow(header)
        for row in results:
            formatted_row = [format_value_for_csv(value, args.decimal_sep) for value in row]
            writer.writerow(formatted_row)

    print(f"Concluído! Features salvas em {args.output}")
    print(f"Total de features por imagem: {feature_size}")


if __name__ == '__main__':
    main()
