import numpy as np
import pandas as pd
import random


df = pd.read_csv("../../../dados/sin5016/diabetes.csv", encoding="utf-8")
print(df.head())

# --- utilidades ---
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def soft_threshold(z, tau):
    # operador proximal do L1 (aplicado elemento a elemento)
    return np.sign(z) * np.maximum(np.abs(z) - tau, 0.0)

def logistic_ce_loss(y_true, p):
    # cross-entropy média (evita log(0))
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean()

# --- treino com Elastic Net via passo proximal ---
def train_logreg_elasticnet(
    X, y, *,
    lr=0.1,          # taxa de aprendizado
    lam=0.1,         # lambda da regularização
    alpha=0.5,       # mistura L1/L2: 0 -> Ridge, 1 -> Lasso
    epochs=1000,
    standardize=True,
    verbose_every=200
):
    """
    X: (n, d)  y: (n,) com rótulos em {0,1}
    Retorna: dict com pesos, bias e estatísticas de normalização
    """
    n, d = X.shape

    # padronização (z-score), exceto se disabled
    if standardize:
        mu = X.mean(axis=0)
        sigma = X.std(axis=0, ddof=0)
        sigma[sigma == 0] = 1.0
        Xs = (X - mu) / sigma
    else:
        mu = np.zeros(d)
        sigma = np.ones(d)
        Xs = X

    # inicialização
    print("Inicializando os pesos...")
    w = np.random.randint(-2, 2, d)
    b = random.randint(-2, 2)

    for t in range(1, epochs + 1):
        z = Xs @ w + b
        p = sigmoid(z)

        # gradiente da perda logística (sem regularização)
        grad_loss_w = (Xs.T @ (p - y)) / n
        grad_loss_b = (p - y).mean()

        # passo "liso": perda + L2
        v = w - lr * (grad_loss_w + lam * (1 - alpha) * w)

        # passo proximal do L1 (gera esparsidade)
        w = soft_threshold(v, lr * lam * alpha)

        # bias não é regularizado
        b -= lr * grad_loss_b

        # logging opcional
        if verbose_every and t % verbose_every == 0:
            ce = logistic_ce_loss(y, p)
            reg = lam * ((1 - alpha) * 0.5 * np.sum(w**2) + alpha * np.sum(np.abs(w)))
            J = ce + reg
            pred = (p >= 0.5).astype(int)
            acc = (pred == y).mean()
            print(f"[{t:5d}] J={J:.4f} CE={ce:.4f} Reg={reg:.4f} Acc={acc:.3f}")

    return {
        "w": w, "b": b,
        "mu": mu, "sigma": sigma,
        "lam": lam, "alpha": alpha
    }

def predict_proba(X, model):
    Xs = (X - model["mu"]) / model["sigma"]
    return sigmoid(Xs @ model["w"] + model["b"])

def predict(X, model, threshold=0.5):
    return (predict_proba(X, model) >= threshold).astype(int)

# --- exemplo de uso (gera dados sintéticos só para demonstrar) ---
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 600
    d = 5

    # data (as pandas dataframes)
    # X = cdc_diabetes_health_indicators.data.features
    X = df[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]]
    y = df["Outcome"]

    # # metadata
    # print(cdc_diabetes_health_indicators.metadata)

    # # variable information
    # print(cdc_diabetes_health_indicators.variables)

    model = train_logreg_elasticnet(
        X, y,
        lr=0.2,
        lam=0.05,
        alpha=0.5,
        epochs=1500,
        verbose_every=300
    )

    p = predict_proba(X, model)
    y_hat = (p >= 0.5).astype(int)
    acc = (y_hat == y).mean()
    print("\nPesos aprendidos:", model["w"])
    print("Bias aprendido  :", model["b"])
    print(f"Acurácia final  : {acc:.3f}")
