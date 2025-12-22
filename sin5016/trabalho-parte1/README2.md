## Classificador Softmax Multiclasse

O projeto conta com um pipeline completo (`svm_softmax/`) para treinar um
classificador softmax multiclasses inspirado em SVM. Ele aceita até 12 000
classes, opera em mini-batches, paraleliza o cálculo de gradientes em CPU e pode
usar GPU via CuPy (opcional).

### Camadas principais

- **Backend (`svm_softmax/backend.py`)** – escolhe automaticamente NumPy (CPU) ou
  CuPy (GPU). Basta passar `--use-gpu` na CLI para tentar usar a GPU (instale a
  versão do CuPy compatível com sua placa, ex.: `cupy-cuda11x`).
- **Kernels (`svm_softmax/kernels.py`)** – encapsula transformações de features.
  Inclui `LinearFeatureMap` e uma aproximação RBF via Random Fourier Features; é
  simples implementar novos kernels sem tocar no restante do código.
- **Fontes de dados (`svm_softmax/data_sources/*`)** – camada intercambiável de
  dados. `InMemoryCSVDataSource` lê o CSV inteiro (modo simples) e
  `StreamingCSVDataSource` mantém apenas buffers em RAM, lendo do disco à medida
  que os batches são solicitados.
- **Modelo (`svm_softmax/model.py`)** – classificador softmax com perda
  cross-entropy, regularização L2, paralelismo via threads (CPU) e suporte a GPU.

### Treinando via CLI

Use `train_softmax_classifier.py` para treinar a partir de um CSV que contenha
uma coluna de rótulos e as colunas de features HOG:

```bash
python train_softmax_classifier.py \
  --csv treino.csv \
  --label-column 1 \
  --feature-slice 2 0 \
  --ignored-columns 0 \
  --data-source streaming \
  --batch-size 512 \
  --epochs 15 \
  --num-workers 8 \
  --kernel linear \
  --use-gpu
```

Principais argumentos:

- `--data-source`: `in-memory` (carrega tudo) ou `streaming` (lê em buffers).
- `--feature-slice`/`--feature-columns`: escolhe as colunas numéricas; combine
  com `--ignored-columns` para descartar, por exemplo, `image_path`.
- `--delimiter`/`--decimal-sep`: ajusta o caractere separador do CSV e o
  separador decimal ao consumir arquivos exportados em outros formatos
  regionais.
- `--kernel`: `linear` ou `rbf` (Random Fourier Features). Outras opções podem
  ser adicionadas implementando `FeatureMap`.
- `--num-workers`: número de threads para gradientes paralelos (somente CPU).
- `--use-gpu`: ativa CuPy.
- `--save-model`: salva os pesos treinados (`npz`).

### Divisão treino/val/teste e histórico de métricas

- `--train-size`, `--val-size`, `--test-size`: percentuais (que devem somar 1.0)
  usados pelo `sklearn.model_selection.train_test_split` para criar os três
  subconjuntos. Por padrão: 70 % treino, 15 % validação e 15 % teste.
- `--split-seed`: controla a semente usada nas divisões e na iteração dos
  mini-batches subsequentes, garantindo reprodutibilidade.
- `--stratify-split`: preserva a distribuição dos rótulos em todas as
  divisões, minimizando o desbalanceamento entre conjuntos.
- `--history-log`: caminho de saída (JSON) com o histórico por época, contendo
  `loss`/`accuracy` de treino e `val_loss`/`val_accuracy`. Útil para monitorar
  overfitting/underfitting posteriormente ou alimentar notebooks de análise.

> **Importante:** certifique-se de adicionar a coluna de rótulos ao CSV
> exportado pelo HOG antes de iniciar o treinamento.

### Uso programático

```python
from svm_softmax.data_sources import InMemoryCSVDataSource
from svm_softmax.kernels import LinearFeatureMap
from svm_softmax.model import SoftmaxClassifier, TrainingConfig

source = InMemoryCSVDataSource("treino.csv", label_column=1, ignored_columns=[0])
model = SoftmaxClassifier(source.num_features, source.num_classes, feature_map=LinearFeatureMap())
config = TrainingConfig(epochs=5, batch_size=256, num_workers=8)
model.fit(source, config)
model.save("pesos.npz")
```

Para GPU, instale CuPy e passe `use_gpu=True` no construtor/configuração.

## Características
