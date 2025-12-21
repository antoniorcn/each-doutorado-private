# Extrator de Features HOG

Script Python para extrair features HOG (Histogram of Oriented Gradients) de imagens PNG e JPG usando OpenCV.

## Instalação

Instale as dependências necessárias:

```bash
pip install -r requirements.txt
```

## Uso

### Sintaxe Básica

```bash
python extract_hog_features.py --input <caminho_imagens> --output <arquivo_csv>
```

### Parâmetros

- `--input` ou `-i`: Caminho para arquivo(s) de imagem. Suporta wildcards (ex: `"imagens/*.jpg"`)
- `--output` ou `-o`: Nome do arquivo CSV de saída com as features extraídas

### Parâmetros Opcionais (HOG)

Todos os parâmetros HOG podem ser configurados via linha de comando:

- `--win-size WIDTH HEIGHT`: Tamanho da janela para HOG em pixels (padrão: 64 64). Deve ser múltiplo de cell-size.
- `--block-size WIDTH HEIGHT`: Tamanho do bloco para HOG em pixels (padrão: 16 16). Deve ser múltiplo de cell-size e não pode ser maior que win-size.
- `--block-stride WIDTH HEIGHT`: Stride do bloco para HOG em pixels (padrão: 8 8). Deve ser múltiplo de cell-size.
- `--cell-size WIDTH HEIGHT`: Tamanho da célula para HOG em pixels (padrão: 8 8).
- `--nbins N`: Número de bins de orientação para HOG (padrão: 9). Valores comuns: 9, 18, 36.

**Nota:** O script valida automaticamente se os parâmetros são compatíveis entre si (múltiplos apropriados, tamanhos relativos corretos, etc.).

### Parâmetros Opcionais (Wavelet - Redução de Features)

Para reduzir o número de features extraídas, você pode aplicar decomposição wavelet antes do HOG:

- `--wavelet-levels N`: Número de níveis de decomposição wavelet a aplicar antes do HOG (padrão: 0, desabilitado). Cada nível reduz a imagem pela metade, reduzindo significativamente o número de features. Use valores como 1, 2, 3.
- `--wavelet TIPO`: Tipo de wavelet a ser usado (padrão: haar). Opções comuns: `haar`, `db4`, `db8`, `bior2.2`, `coif2`.

**Como funciona:** O algoritmo aplica decomposição wavelet múltiplas vezes, extraindo apenas a sub-banda LL (Low-Low) em cada nível. Isso reduz a resolução da imagem antes de passar para o HOG, resultando em menos features. Por exemplo:
- 1 nível: imagem reduzida para 50% (25% dos pixels)
- 2 níveis: imagem reduzida para 25% (6.25% dos pixels)
- 3 níveis: imagem reduzida para 12.5% (1.56% dos pixels)

### Exemplos

1. Processar todas as imagens JPG em um diretório:
```bash
python extract_hog_features.py --input "imagens/*.jpg" --output features.csv
```

2. Processar todas as imagens PNG:
```bash
python extract_hog_features.py --input "fotos/*.png" --output resultado.csv
```

3. Processar todas as imagens em um diretório:
```bash
python extract_hog_features.py --input "data/" --output features.csv
```

4. Processar um arquivo específico:
```bash
python extract_hog_features.py --input "imagem.jpg" --output features.csv
```

5. Com parâmetros customizados do HOG:
```bash
python extract_hog_features.py --input "*.jpg" --output features.csv --win-size 128 128 --nbins 18
```

6. Configuração completa de parâmetros HOG:
```bash
python extract_hog_features.py --input "imagens/*.png" --output features.csv \
  --win-size 128 128 --block-size 32 32 --block-stride 16 16 --cell-size 16 16 --nbins 18
```

7. Com redução de features usando wavelet (2 níveis):
```bash
python extract_hog_features.py --input "imagens/*.jpg" --output features.csv --wavelet-levels 2
```

8. Com wavelet customizado e parâmetros HOG:
```bash
python extract_hog_features.py --input "*.png" --output features.csv \
  --wavelet-levels 3 --wavelet db4 --win-size 64 64
```

## Formato de Saída

O arquivo CSV gerado contém:
- Primeira coluna: `image_path` - caminho da imagem processada
- Colunas seguintes: `feature_0`, `feature_1`, ..., `feature_N` - valores das features HOG

## Características

- ✅ Suporte completo a type hints (tipagem estática)
- ✅ Validação automática de parâmetros HOG
- ✅ Redução de features usando decomposição wavelet (sub-banda LL)
- ✅ Controle do número de níveis de decomposição wavelet
- ✅ Suporte a wildcards para seleção de múltiplos arquivos
- ✅ Processamento em lote de imagens
- ✅ Mensagens informativas de progresso
- ✅ Tratamento robusto de erros

## Requisitos

- Python 3.6+
- OpenCV (opencv-python)
- NumPy
- PyWavelets (para suporte a decomposição wavelet)

