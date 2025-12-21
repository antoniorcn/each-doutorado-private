from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from itertools import combinations

# ==========================================
# utilitários
# ==========================================
def sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out

def add_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((X.shape[0], 1), dtype=X.dtype), X])

def standardize(X: np.ndarray, mean=None, std=None):
    if mean is None: mean = X.mean(axis=0)
    if std  is None: std  = X.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return (X - mean) / std, mean, std

# ==========================================
# Regressão Logística Binária (GD)
# ==========================================
class LogisticRegressionGD:
    def __init__(
        self,
        lr: Union[float, None] = 0.1,
        epochs: int = 1000,
        l2: float = 0.0,
        batch_size: Union[int, None] = None,
        tol: Union[float, None] = None,
        shuffle: bool = True,
        verbose: bool = False,
        random_state: Union[int, None] = None,
        standardize_features: bool = True,
    ):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.batch_size = batch_size
        self.tol = tol
        self.shuffle = shuffle
        self.verbose = verbose
        self.random_state = random_state
        self.standardize_features = standardize_features

        self.w_ = None
        self.mean_ = None
        self.std_ = None
        self.loss_history_ = []

        if random_state is not None:
            np.random.seed(random_state)

    @staticmethod
    def _log_loss(y_true: np.ndarray, y_prob: np.ndarray, w: np.ndarray, l2: float) -> float:
        eps = 1e-12
        y_prob = np.clip(y_prob, eps, 1.0 - eps)
        ce = - (y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)).mean()
        if l2 > 0.0:
            ce += 0.5 * l2 * np.dot(w[1:], w[1:])
        return ce

    def _gradient(self, Xb: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        n = Xb.shape[0]
        p = sigmoid(Xb @ w)
        grad = (Xb.T @ (p - y)) / n
        if self.l2 > 0.0:
            reg = np.r_[0.0, w[1:]]
            grad += self.l2 * reg
        return grad

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)

        if self.standardize_features:
            X, self.mean_, self.std_ = standardize(X)
        else:
            self.mean_, self.std_ = None, None

        Xb = add_bias(X)
        n_features = Xb.shape[1]
        if self.w_ is None:
            self.w_ = np.random.randn(n_features) * 0.01

        batch_size = self.batch_size or Xb.shape[0]
        last_loss = np.inf
        self.loss_history_.clear()

        for epoch in range(self.epochs):
            idx = np.arange(Xb.shape[0])
            if self.shuffle and batch_size < Xb.shape[0]:
                np.random.shuffle(idx)

            for start in range(0, Xb.shape[0], batch_size):
                mb = idx[start:start+batch_size]
                grad = self._gradient(Xb[mb], y[mb], self.w_)
                self.w_ -= self.lr * grad

            # loss por época
            y_prob = sigmoid(Xb @ self.w_)
            loss = self._log_loss(y, y_prob, self.w_, self.l2)
            self.loss_history_.append(loss)

            if self.tol is not None:
                if abs(last_loss - loss) < self.tol:
                    break
                last_loss = loss

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.standardize_features:
            X, _, _ = standardize(X, self.mean_, self.std_)
        Xb = add_bias(X)
        return sigmoid(Xb @ self.w_)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int64)
    
    def decision(self, X: np.ndarray) -> np.ndarray:
        """logit = Xb @ w (margem, útil p/ desempate)."""
        X = np.asarray(X, dtype=np.float64)
        if self.standardize_features:
            X, _, _ = standardize(X, self.mean_, self.std_)
        Xb = add_bias(X)
        return Xb @ self.w_

# =========================
# One-vs-One (OvO) wrapper
# =========================
class LogisticRegressionOvO:
    """
    Multiclasse via One-vs-One: treina um classificador binário para cada par (a,b).
    Predição por maioria de votos; empate desempatado pela soma de margens (logits).
    """
    def __init__(
        self,
        lr: float = 0.1,
        epochs: int = 1000,
        l2: float = 0.0,
        batch_size: Union[int, None] = None,
        tol: Union[float, None] = None,
        shuffle: bool = True,
        verbose: bool = False,
        random_state: Union[int, None] = None,
        standardize_features: bool = True,
        shared_scaler: bool = True,
    ):
        self.params_ = dict(
            lr=lr, epochs=epochs, l2=l2, batch_size=batch_size, tol=tol,
            shuffle=shuffle, verbose=verbose, random_state=random_state,
            standardize_features=False  # normalização será gerenciada aqui
        )
        self.standardize_features = standardize_features
        self.shared_scaler = shared_scaler

        self.classes_ = None
        # dict[(a,b)] -> (model, positivo=a, negativo=b)
        self.models_: dict[tuple, LogisticRegressionGD] = {}

        self.mean_ = None
        self.std_ = None

    def _maybe_standardize(self, X: np.ndarray, fit=False):
        if not self.standardize_features:
            return X
        if fit:
            Xn, self.mean_, self.std_ = standardize(X)
            return Xn
        else:
            return standardize(X, self.mean_, self.std_)[0]

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).reshape(-1)
        self.classes_ = np.unique(y)

        # normalização compartilhada (opcional)
        Xn = self._maybe_standardize(X, fit=True)

        self.models_.clear()
        for a, b in combinations(self.classes_, 2):
            # seleciona apenas as amostras das classes a e b
            mask = (y == a) | (y == b)
            Xa = Xn[mask] if self.shared_scaler else X[mask]
            ya = (y[mask] == a).astype(np.float64)  # 1 para a, 0 para b

            model = LogisticRegressionGD(**self.params_)
            model.standardize_features = False if self.shared_scaler else self.standardize_features
            model.fit(Xa, ya)

            self.models_[(a, b)] = model

        return self

    def _votes_and_margins(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        Xn = self._maybe_standardize(X, fit=False)
        n, K = X.shape[0], len(self.classes_)
        votes = np.zeros((n, K), dtype=int)
        margins = np.zeros((n, K), dtype=np.float64)  # para desempate

        # índice da classe para colunas
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        for (a, b), model in self.models_.items():
            # entrada para o modelo: precisa estar no mesmo regime usado no fit
            Xin = Xn if self.shared_scaler else X

            # probabilidade de "a" (label positivo)
            p_a = model.predict_proba(Xin)
            # decisão: a se p>=0.5; senão b
            pred_a = (p_a >= 0.5)

            ia, ib = class_to_idx[a], class_to_idx[b]
            votes[np.where(pred_a), ia] += 1
            votes[np.where(~pred_a), ib] += 1

            # margem (logit), positiva favorece "a", negativa favorece "b"
            logit = model.decision(Xin)
            margins[:, ia] += np.maximum(logit, 0.0)
            margins[:, ib] += np.maximum(-logit, 0.0)

        return votes, margins

    def predict(self, X: np.ndarray) -> np.ndarray:
        votes, margins = self._votes_and_margins(X)
        # vencedor por votos; se empatar, usa maior margem acumulada
        max_votes = votes.max(axis=1, keepdims=True)
        tied = votes == max_votes
        # para cada linha, dentre os empatados escolhe o de maior margem
        winners = []
        for i in range(votes.shape[0]):
            candidates = np.where(tied[i])[0]
            j = candidates[np.argmax(margins[i, candidates])]
            winners.append(self.classes_[j])
        return np.asarray(winners)

    def predict_vote_proba(self, X: np.ndarray) -> np.ndarray:
        """
        'Probabilidade' via proporção de votos (não calibrada; soma 1).
        Útil só como score simplificado.
        """
        votes, _ = self._votes_and_margins(X)
        total = votes.sum(axis=1, keepdims=True)  # = nº de pares = K*(K-1)/2
        total = np.maximum(total, 1)
        return votes / total

# ==========================================
# Exemplo com 3 classes
# ==========================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    df = pd.read_csv("Iris.csv", encoding="utf-8")

    # 3 blobs 2D
    n = df.shape[0]
    cov = np.array([[1.0, 0.3],[0.3, 1.0]])
    X0 = df["SepalLengthCm"]
    X1 = df["SepalWidthCm"]
    X2 = df["PetalLengthCm"]
    X3 = df["PetalWidthCm"]
    X = np.transpose(np.array([X0, X1, X2, X3]))
    df["output"] = df["Species"].replace({
        "Iris-setosa": 0, 
        "Iris-versicolor": 1, 
        "Iris-virginica": 2 }, inplace=False)
    y = df["output"]

    print(df.head())

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # treinar OvA
    ovo = LogisticRegressionOvO(
        lr=0.1,
        epochs=2000,
        l2=1e-2,
        batch_size=64,
        tol=1e-8,
        shuffle=True,
        verbose=False,
        random_state=42,
        standardize_features=True,
        shared_scaler=True,   # uma normalização para todos os K modelos
    ).fit(X_tr, y_tr)

    # avaliação
    y_pred = ovo.predict(X_te)
    acc = (y_pred == y_te).mean()

    # matriz de confusão simples (NumPy puro)
    K = len(np.unique(y))
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(y_te, y_pred):
        cm[t, p] += 1

    print("Acurácia:", acc)
    print("Matriz de confusão (linhas=verdadeiro, colunas=previsto):\n", cm)
    print("Proporções de votos (primeiras 5 amostras):\n",
          np.round(ovo.predict_vote_proba(X_te[:5]), 3))

