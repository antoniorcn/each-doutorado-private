import numpy as np

# ---------------------------
# utilitários numéricos
# ---------------------------
def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Sigmóide estável numericamente.
    Evita overflow quando |z| é grande.
    """
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out

def add_bias(X: np.ndarray) -> np.ndarray:
    """Adiciona coluna de 1s (termo de viés)."""
    return np.hstack([np.ones((X.shape[0], 1), dtype=X.dtype), X])

def standardize(X: np.ndarray, mean=None, std=None):
    """
    Normaliza features: (X - mean) / std.
    Retorna X_norm, mean, std.
    std mínima forçada para evitar divisão por zero.
    """
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return (X - mean) / std, mean, std

# ---------------------------
# Modelo: Regressão Logística GD
# ---------------------------
class LogisticRegressionGD:
    def __init__(
        self,
        lr: float = 0.1,
        epochs: int = 1000,
        l2: float = 0.0,
        batch_size: int | None = None,
        tol: float | None = None,
        shuffle: bool = True,
        verbose: bool = False,
        random_state: int | None = None,
        standardize_features: bool = True,
    ):
        """
        lr: taxa de aprendizado
        epochs: número de épocas
        l2: força da regularização L2 (não penaliza o viés)
        batch_size: tamanho do mini-batch; None => batch completo
        tol: critério de parada antecipada pela variação de loss (ex.: 1e-6)
        shuffle: embaralhar dados a cada época (para mini-batch)
        standardize_features: normalizar X antes do treino
        """
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.batch_size = batch_size
        self.tol = tol
        self.shuffle = shuffle
        self.verbose = verbose
        self.random_state = random_state
        self.standardize_features = standardize_features

        # aprendidos no fit
        self.w_ = None             # pesos (inclui viés)
        self.mean_ = None          # média das features
        self.std_ = None           # desvio-padrão das features
        self.loss_history_ = []    # histórico de loss

        if random_state is not None:
            np.random.seed(random_state)

    @staticmethod
    def _log_loss(y_true: np.ndarray, y_prob: np.ndarray, w: np.ndarray, l2: float) -> float:
        """
        Binary cross-entropy com L2 (sem penalizar viés).
        y_true em {0,1}; y_prob em (0,1).
        """
        eps = 1e-12
        y_prob = np.clip(y_prob, eps, 1.0 - eps)
        ce = - (y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)).mean()

        if l2 > 0.0:
            # não penaliza w[0] (viés)
            ce += 0.5 * l2 * np.dot(w[1:], w[1:])
        return ce

    def _gradient(self, Xb: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Gradiente da perda (BCE + L2) em relação a w.
        Xb já inclui a coluna do viés.
        """
        n = Xb.shape[0]
        z = Xb @ w
        p = sigmoid(z)
        # grad BCE
        grad = (Xb.T @ (p - y)) / n

        # L2 (sem viés)
        if self.l2 > 0.0:
            reg = np.r_[0.0, w[1:]]  # zero no viés
            grad += self.l2 * reg
        return grad

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Treina o modelo.
        X: (n_amostras, n_features)
        y: vetor binário {0,1} com shape (n_amostras,)
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)

        # normalização (opcional)
        if self.standardize_features:
            X, self.mean_, self.std_ = standardize(X)
        else:
            self.mean_, self.std_ = None, None

        Xb = add_bias(X)  # adiciona viés
        n_features = Xb.shape[1]
        # inicialização de pesos
        if self.w_ is None:
            # pequena aleatoriedade ajuda a quebrar simetria
            self.w_ = np.random.randn(n_features) * 0.01

        # define tamanho do batch
        batch_size = self.batch_size or Xb.shape[0]

        last_loss = np.inf
        self.loss_history_.clear()

        for epoch in range(self.epochs):
            # embaralhar índices
            idx = np.arange(Xb.shape[0])
            if self.shuffle and batch_size < Xb.shape[0]:
                np.random.shuffle(idx)

            # laço de mini-batches
            for start in range(0, Xb.shape[0], batch_size):
                end = start + batch_size
                mb_idx = idx[start:end]
                X_mb = Xb[mb_idx]
                y_mb = y[mb_idx]

                grad = self._gradient(X_mb, y_mb, self.w_)
                self.w_ -= self.lr * grad

            # avalia perda por época (no conjunto inteiro)
            y_prob = sigmoid(Xb @ self.w_)
            loss = self._log_loss(y, y_prob, self.w_, self.l2)
            self.loss_history_.append(loss)

            if self.verbose and (epoch % max(1, self.epochs // 10) == 0 or epoch == self.epochs - 1):
                print(f"[época {epoch+1}/{self.epochs}] loss={loss:.6f}")

            # parada antecipada
            if self.tol is not None:
                if abs(last_loss - loss) < self.tol:
                    if self.verbose:
                        print(f"Parada antecipada na época {epoch+1}: Δloss={abs(last_loss - loss):.3e} < tol={self.tol}")
                    break
                last_loss = loss

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna probabilidade da classe 1.
        """
        X = np.asarray(X, dtype=np.float64)
        if self.standardize_features:
            X, _, _ = standardize(X, self.mean_, self.std_)
        Xb = add_bias(X)
        return sigmoid(Xb @ self.w_)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predição binária {0,1}.
        """
        return (self.predict_proba(X) >= threshold).astype(np.int64)

# ---------------------------
# Exemplo de uso (dados sintéticos)
# ---------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Gera duas classes gaussianas em 2D
    n0 = 300
    n1 = 300
    cov = np.array([[1.0, 0.4],
                    [0.4, 1.0]])
    X0 = rng.multivariate_normal(mean=[-1.5, -1.0], cov=cov, size=n0)
    X1 = rng.multivariate_normal(mean=[+1.2, +1.5], cov=cov, size=n1)
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n0), np.ones(n1)])

    # embaralha e separa em treino/teste (80/20)
    idx = rng.permutation(len(X))
    X = X[idx]; y = y[idx]
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test,  y_test  = X[split:], y[split:]

    # treina
    clf = LogisticRegressionGD(
        lr=0.1,
        epochs=2000,
        l2=1e-2,                # experimente 0.0, 1e-3, 1e-2, 1e-1…
        batch_size=64,          # None => batch completo; <N => mini-batch
        tol=1e-8,               # parada antecipada
        shuffle=True,
        verbose=True,
        random_state=42,
        standardize_features=True,
    ).fit(X_train, y_train)

    # avalia
    prob = clf.predict_proba(X_test)
    y_pred = (prob >= 0.5).astype(int)
    acc = (y_pred == y_test).mean()

    # métricas simples
    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    tn = int(((y_pred == 0) & (y_test == 0)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())

    precision = tp / (tp + fp + 1e-12)
    recall    = tp / (tp + fn + 1e-12)
    f1        = 2 * precision * recall / (precision + recall + 1e-12)

    print("\nResultados no teste")
    print(f"Acurácia: {acc:.4f}")
    print(f"Precisão: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
    print(f"Loss final (treino): {clf.loss_history_[-1]:.6f}")
