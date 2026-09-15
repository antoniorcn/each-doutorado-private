# Instruções do projeto — Denoising de sinais EEG com aprendizado profundo

## 1. Visão geral

Este projeto tem como objetivo desenvolver e avaliar modelos de aprendizado profundo para remoção de ruídos artificiais adicionados a sinais de eletroencefalograma provenientes da base TUEV.

O projeto terá inicialmente dois modelos principais:

1. uma U-Net 1D supervisionada, utilizada como baseline;
2. uma GAN 1D, formada por um gerador baseado em U-Net residual e um discriminador temporal.

O modelo receberá um segmento de EEG artificialmente contaminado e deverá reconstruir o respectivo sinal de referência pré-processado.

O sinal de referência não deve ser chamado de “EEG clinicamente limpo”, pois a base TUEV contém sinais clínicos reais que podem possuir artefatos naturais. O termo correto no projeto deve ser:

* sinal de referência pré-processado;
* clean reference apenas como nomenclatura computacional interna;
* sinal artificialmente corrompido para a entrada do modelo.

## 2. Objetivo científico

Investigar se modelos de aprendizado profundo conseguem remover ruídos artificiais de sinais EEG preservando:

* morfologia temporal;
* amplitude;
* fase;
* conteúdo espectral;
* eventos transitórios;
* relações entre canais;
* características relevantes para classificação clínica.

A qualidade do resultado não pode ser avaliada apenas visualmente. O projeto deve utilizar métricas temporais, espectrais e, futuramente, o desempenho de um classificador downstream.

## 3. Hipótese experimental

A hipótese principal é que um gerador temporal treinado com perdas de reconstrução e perdas espectrais pode recuperar sinais EEG contaminados artificialmente.

Uma segunda hipótese é que a inclusão de treinamento adversarial pode tornar os sinais reconstruídos mais semelhantes à distribuição dos sinais de referência.

A GAN deve sempre ser comparada com modelos e métodos mais simples. O treinamento adversarial não deve ser considerado automaticamente superior.

## 4. Tecnologias

Utilizar preferencialmente:

* Python 3.11;
* PyTorch;
* MNE-Python;
* NumPy;
* SciPy;
* pandas;
* scikit-learn;
* matplotlib;
* PyYAML ou Hydra para configurações;
* pytest para testes;
* TensorBoard ou Weights & Biases para monitoramento.

Evitar dependências desnecessárias.

Todo código deve possuir type hints, docstrings, validação de argumentos e tratamento explícito de erros.

## 5. Pipeline geral

O fluxo será:

1. localizar arquivos EDF da base TUEV;
2. ler os sinais com MNE;
3. normalizar os nomes dos canais;
4. selecionar uma montagem comum;
5. definir ou converter a referência;
6. aplicar filtros;
7. padronizar a frequência de amostragem;
8. dividir os registros em segmentos;
9. normalizar amplitudes;
10. tratar ou limitar valores extremos;
11. criar divisões por paciente;
12. adicionar ruídos artificiais;
13. armazenar pares de entrada e referência;
14. treinar o baseline;
15. treinar a GAN;
16. avaliar e comparar os modelos.

## 6. Pré-processamento

O pré-processamento inicial deve permitir configuração dos seguintes parâmetros:

* frequência de amostragem alvo;
* frequência inferior do filtro passa-faixa;
* frequência superior do filtro passa-faixa;
* frequência do filtro notch;
* duração das janelas;
* sobreposição entre janelas;
* lista de canais obrigatórios;
* estratégia de referência;
* estratégia de normalização;
* limite de amplitudes extremas;
* política para registros com canais ausentes.

Valores iniciais sugeridos:

* frequência de amostragem: 200 Hz;
* filtro passa-faixa: 0,5 a 45 Hz;
* notch: configurável entre 50 e 60 Hz;
* janela: 5 segundos;
* sobreposição inicial: zero;
* referência: média dos canais selecionados.

Esses valores devem estar em arquivo de configuração, e não fixados diretamente no código.

## 7. Canais

Os nomes de canais podem aparecer em formatos diferentes, como:

* FP1;
* EEG FP1-REF;
* FP1-LE;
* FP1-AVG.

Criar uma função responsável por:

1. converter o nome para letras maiúsculas;
2. remover prefixos como EEG;
3. remover sufixos de referência;
4. remover espaços desnecessários;
5. mapear nomenclaturas equivalentes;
6. detectar canais duplicados;
7. registrar canais ausentes.

Não assumir silenciosamente que todos os arquivos possuem os mesmos canais.

Uma lista inicial de canais candidatos é:

* FP1;
* FP2;
* F3;
* F4;
* C3;
* C4;
* P3;
* P4;
* O1;
* O2;
* F7;
* F8;
* T3 ou T7;
* T4 ou T8;
* T5 ou P7;
* T6 ou P8;
* FZ;
* CZ;
* PZ.

O mapeamento entre nomenclaturas antigas e atuais deve ser explícito e testável.

## 8. Segmentação

Cada segmento deve possuir o formato:

```
[número_de_canais, número_de_amostras]
```

Para 19 canais, 200 Hz e janelas de 5 segundos:

```
[19, 1000]
```

O dataset PyTorch deve retornar pelo menos:

* noisy_signal;
* reference_signal;
* patient_id;
* recording_id;
* segment_start;
* sampling_rate;
* noise_metadata.

A divisão entre treino, validação e teste deve ser feita por paciente.

É proibido colocar segmentos do mesmo paciente em mais de uma partição.

## 9. Normalização

Implementar estratégias configuráveis:

* z-score por canal e segmento;
* z-score por canal usando estatísticas do treinamento;
* normalização robusta usando mediana e intervalo interquartil;
* escalonamento por amplitude de referência.

Evitar vazamento de dados.

Quando estatísticas globais forem utilizadas, elas devem ser calculadas exclusivamente no conjunto de treinamento e aplicadas aos conjuntos de validação e teste.

Guardar os parâmetros de normalização junto com os checkpoints.

## 10. Tratamento de valores extremos

A limitação de amplitudes não deve ser aplicada silenciosamente.

Implementar pelo menos duas estratégias:

* clipping por limite físico configurável;
* winsorization ou clipping por percentis calculados no conjunto de treinamento.

Registrar:

* quantidade de amostras alteradas;
* canais afetados;
* limites utilizados;
* arquivos ou segmentos descartados.

Não eliminar automaticamente espículas ou eventos raros, pois eles podem possuir relevância clínica.

## 11. Geração de ruído artificial

Criar um pipeline modular de corrupção.

Cada tipo de ruído deve ser implementado como um componente independente e receber um gerador aleatório controlado por seed.

Tipos iniciais:

### 11.1 Ruído gaussiano

Adicionar ruído gaussiano com intensidade definida por:

* desvio-padrão relativo;
* SNR alvo;
* intensidade sorteada dentro de um intervalo configurável.

### 11.2 Interferência elétrica

Adicionar senoides de:

* 50 Hz;
* 60 Hz;
* harmônicas opcionais.

Variar:

* amplitude;
* fase;
* duração;
* canais atingidos.

### 11.3 Deriva de linha de base

Adicionar componentes lentas, aproximadamente entre 0,05 e 0,5 Hz.

### 11.4 Artefato muscular sintético

Gerar ruído de maior frequência, aplicado a intervalos e canais aleatórios.

### 11.5 Artefato ocular sintético

Gerar pulsos lentos de maior amplitude, preferencialmente mais fortes nos canais frontais.

### 11.6 Electrode pop

Simular transientes abruptos associados a alteração momentânea do contato do eletrodo.

### 11.7 Saturação e dropout

Simular:

* clipping;
* trechos constantes;
* trechos zerados;
* perda parcial de um canal.

Cada amostra deve guardar metadados indicando:

* tipos de ruído aplicados;
* parâmetros sorteados;
* canais afetados;
* SNR antes e depois;
* seed utilizada.

## 12. Reprodutibilidade

Toda aleatoriedade deve ser controlável por seed.

Controlar pelo menos:

* Python random;
* NumPy;
* PyTorch;
* CUDA, quando aplicável;
* DataLoader workers;
* geração de ruídos;
* divisão de pacientes.

Registrar:

* versão das bibliotecas;
* parâmetros;
* commit do Git;
* hardware;
* seed;
* data do experimento.

Não prometer reprodução bit a bit entre diferentes GPUs ou versões de CUDA.

## 13. Baseline U-Net 1D

Antes da GAN, implementar uma U-Net 1D supervisionada.

Entrada:

```
[batch, canais, amostras]
```

Saída:

```
[batch, canais, amostras]
```

A arquitetura deve possuir:

* blocos Conv1D;
* normalização adequada;
* função de ativação configurável;
* downsampling;
* bottleneck;
* upsampling;
* skip connections;
* projeção final com o mesmo número de canais da entrada.

Evitar alterar desnecessariamente o comprimento temporal.

Validar automaticamente se a saída possui a mesma dimensão da entrada.

## 14. GAN 1D

A GAN será composta por:

### Gerador

Utilizar inicialmente a mesma U-Net 1D do baseline.

O gerador deve reconstruir o sinal de referência a partir do sinal artificialmente contaminado.

### Discriminador

Implementar um discriminador temporal 1D, preferencialmente no estilo PatchGAN.

O discriminador deverá avaliar pares condicionais:

```
sinal ruidoso + sinal de referência
```

e:

```
sinal ruidoso + sinal gerado
```

O treinamento deve considerar uma GAN condicional, pois o objetivo não é gerar qualquer EEG realista, mas reconstruir o EEG correspondente à entrada.

## 15. Funções de perda

O gerador não deve utilizar apenas perda adversarial.

Implementar inicialmente:

### Perda temporal

* L1;
* opcionalmente MSE para comparação.

### Perda espectral

Comparar magnitude espectral por meio de FFT ou STFT.

A implementação deve evitar instabilidades numéricas e tratar adequadamente escala logarítmica.

### Perda de correlação

Avaliar preservação da forma temporal por canal.

### Perda adversarial

Implementar uma opção estável, como:

* least-squares GAN;
* hinge loss;
* Wasserstein GAN com gradient penalty.

Não implementar todas simultaneamente no primeiro estágio.

Começar com uma alternativa simples e testável.

A perda total deve possuir pesos configuráveis, por exemplo:

```
generator_loss =
    lambda_time * temporal_loss
    + lambda_frequency * spectral_loss
    + lambda_adversarial * adversarial_loss
    + lambda_correlation * correlation_loss
```

Nunca ocultar os valores dos componentes individuais da perda.

## 16. Métricas

Calcular no mínimo:

* MAE;
* MSE;
* RMSE;
* correlação de Pearson;
* SNR de entrada;
* SNR de saída;
* melhoria de SNR;
* erro de magnitude espectral;
* potência absoluta por banda;
* potência relativa por banda.

Bandas iniciais:

* delta;
* theta;
* alfa;
* beta;
* gama, limitada pela frequência de corte utilizada.

As métricas devem ser calculadas:

* por segmento;
* por canal;
* por paciente;
* globalmente.

Apresentar média, mediana, desvio-padrão e intervalos de confiança quando apropriado.

## 17. Validação qualitativa

Criar funções para visualizar:

* sinal de referência;
* sinal contaminado;
* sinal reconstruído;
* erro residual;
* espectro de frequência;
* espectrograma;
* métricas do segmento.

Os gráficos devem mostrar unidades e frequência de amostragem.

Não selecionar apenas exemplos favoráveis.

Permitir amostragem aleatória reproduzível de resultados.

## 18. Segurança científica

Modelos adversariais podem criar padrões plausíveis inexistentes ou remover eventos importantes.

Por isso:

* não afirmar que o modelo produz EEG clinicamente limpo;
* não utilizar somente avaliação visual;
* não considerar realismo como sinônimo de fidelidade;
* não suavizar automaticamente eventos transitórios;
* comparar sempre com o sinal de referência;
* avaliar preservação espectral e temporal;
* registrar limitações;
* não apresentar o sistema como dispositivo médico.

O projeto é experimental e acadêmico.

## 19. Estrutura esperada

Organizar inicialmente o repositório como:

```
eeg_gan_denoising/
├── .github/
│   └── copilot-instructions.md
├── configs/
│   └── base.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── splits/
├── src/
│   └── eeg_denoising/
│       ├── __init__.py
│       ├── config.py
│       ├── preprocessing/
│       │   ├── channels.py
│       │   ├── filters.py
│       │   ├── normalization.py
│       │   ├── segmentation.py
│       │   └── tuev.py
│       ├── corruption/
│       │   ├── base.py
│       │   ├── gaussian.py
│       │   ├── powerline.py
│       │   ├── baseline.py
│       │   ├── ocular.py
│       │   ├── muscle.py
│       │   └── pipeline.py
│       ├── datasets/
│       │   └── paired_eeg.py
│       ├── models/
│       │   ├── blocks.py
│       │   ├── unet.py
│       │   ├── discriminator.py
│       │   └── gan.py
│       ├── losses/
│       │   ├── temporal.py
│       │   ├── spectral.py
│       │   └── adversarial.py
│       ├── metrics/
│       │   ├── temporal.py
│       │   ├── spectral.py
│       │   └── snr.py
│       ├── training/
│       │   ├── baseline_trainer.py
│       │   └── gan_trainer.py
│       └── visualization/
│           └── plots.py
├── scripts/
│   ├── inspect_tuev.py
│   ├── preprocess_tuev.py
│   ├── train_baseline.py
│   ├── train_gan.py
│   └── evaluate.py
├── tests/
├── pyproject.toml
├── README.md
└── .gitignore
```

## 20. Regras de implementação

Ao gerar código:

1. não criar arquivos gigantes;
2. manter responsabilidades separadas;
3. utilizar pathlib em vez de manipulação manual de caminhos;
4. utilizar logging em vez de prints dispersos;
5. validar formatos e dimensões;
6. documentar unidades;
7. nunca presumir que os sinais estão em microvolts sem verificar;
8. impedir vazamento entre pacientes;
9. não baixar automaticamente a base de dados;
10. não colocar caminhos locais absolutos no código;
11. não armazenar credenciais;
12. não alterar sinais silenciosamente;
13. incluir testes para funções críticas;
14. preferir funções puras no pré-processamento;
15. manter configurações fora dos módulos de implementação;
16. não implementar etapas futuras antes de estabilizar a etapa atual.

## 21. Estratégia incremental

Desenvolver nesta ordem:

### Etapa 1

Criar estrutura do projeto, configuração, logging e testes.

### Etapa 2

Criar um script que inspecione arquivos EDF e produza relatório de:

* canais;
* frequência de amostragem;
* duração;
* unidades;
* anotações;
* pacientes;
* inconsistências.

### Etapa 3

Implementar normalização dos nomes dos canais.

### Etapa 4

Implementar filtragem, reamostragem e segmentação.

### Etapa 5

Criar divisão por paciente.

### Etapa 6

Implementar um único ruído sintético gaussiano com SNR controlada.

### Etapa 7

Criar o Dataset PyTorch pareado.

### Etapa 8

Treinar uma U-Net 1D pequena em um subconjunto.

### Etapa 9

Implementar métricas e visualização.

### Etapa 10

Adicionar gradualmente outros tipos de ruído.

### Etapa 11

Adicionar o discriminador e o treinamento adversarial.

### Etapa 12

Realizar experimentos comparativos.

## 22. Instrução ao assistente de programação

Antes de gerar uma implementação extensa:

1. examine os arquivos existentes;
2. respeite a arquitetura atual;
3. descreva resumidamente os arquivos que serão criados ou alterados;
4. implemente apenas a etapa solicitada;
5. inclua testes;
6. informe suposições;
7. destaque riscos de vazamento de dados;
8. destaque alterações que possam modificar a morfologia do EEG;
9. não invente detalhes da TUEV sem confirmação nos arquivos ou na documentação;
10. não declare que o código está correto sem executar ou testar.

Quando houver dúvida entre uma solução sofisticada e uma solução simples, priorize primeiro a solução simples, verificável e reproduzível.
