%pip install -q --upgrade pip
%pip install -q pennylane amazon-braket-pennylane-plugin scikit-learn matplotlib

import pennylane as qml
from pennylane import numpy as np

# SV1 sense especificar bucket: Braket farà servir el bucket per defecte
SV1_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
dev = qml.device(
    "braket.aws.qubit",
    device_arn=SV1_ARN,
    wires=4,
    shots=None
)

print(dev)

def make_synthetic_embeddings(n=240, dim=4, seed=42):
    rng = np.random.default_rng(seed)
    X_good = rng.normal(loc=0.0, scale=0.8, size=(n//2, dim))
    X_bad  = rng.normal(loc=0.0, scale=0.8, size=(n//2, dim))
    # simula un desplaçament de la distribució "maliciosa" (p. ex. patró de prompt-injection)
    X_bad[:, :2] += 2.0
    X = np.vstack([X_good, X_bad])
    y = np.hstack([np.zeros(n//2), np.ones(n//2)])
    # escala a rang d’angles raonable ([-π/2, π/2])
    X = X / (np.max(np.abs(X), axis=0) + 1e-8) * (np.pi/2)
    return X, y

X, y = make_synthetic_embeddings(n=240, dim=4, seed=42)

# Train/Test split
idx = np.random.permutation(len(X))
train_idx, test_idx = idx[:180], idx[180:]
Xtr, ytr = X[train_idx], y[train_idx]
Xte, yte = X[test_idx], y[test_idx]

X.shape, y.shape

n_qubits = 4
layers = 3

def feature_map(x):
    # codifica el vector en rotacions Y (AngleEmbedding)
    qml.templates.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")

def variational_block(params):
    # 3 capes: rotacions locals + entrellaçament en anell (CZ)
    for layer in params:
        for w in range(n_qubits):
            qml.RY(layer[w], wires=w)
        for w in range(n_qubits):
            qml.CZ(wires=[w, (w+1) % n_qubits])

# Diff method:
# - SV1 pot aprofitar mètodes eficients (p. ex. adjoint) per a gradients en QML
#   (si canvies a QPU, el plugin farà servir estratègies adequades)
#   Vegeu també el post sobre adjoint a Braket + PennyLane.
@qml.qnode(dev, interface="autograd", diff_method="best")
def q_model(x, params):
    feature_map(x)
    variational_block(params)
    return qml.expval(qml.PauliZ(0))  # sortida en [-1, 1]

def predict_proba(x, params):
    return 0.5 * (1 - q_model(x, params))  # mapeig a [0, 1]

# Inicialitza paràmetres
params = 0.01 * np.random.randn(layers, n_qubits)

def bce_loss(params, Xb, yb):
    preds = np.array([predict_proba(x, params) for x in Xb])
    eps = 1e-7
    return -np.mean(yb*np.log(preds+eps) + (1-yb)*np.log(1-preds+eps))

opt = qml.AdamOptimizer(stepsize=0.05)
epochs = 60
batch = 32

loss_hist = []
for epoch in range(epochs):
    sel = np.random.choice(len(Xtr), size=batch, replace=False)
    Xb, yb = Xtr[sel], ytr[sel]
    params = opt.step(lambda p: bce_loss(p, Xb, yb), params)
    if (epoch+1) % 10 == 0:
        L = float(bce_loss(params, Xtr, ytr))
        loss_hist.append(L)
        print(f"Epoch {epoch+1:02d}  Train BCE: {L:.4f}")

# 📊 Imports necessaris per a les mètriques
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 🔎 Avaluació sobre el conjunt de test
probs = np.array([predict_proba(x, params) for x in Xte])
preds = (probs >= 0.5).astype(int)

acc = accuracy_score(yte, preds)
cm = confusion_matrix(yte, preds)

print("Test accuracy:", round(acc, 4))
print("Confusion matrix:\n", cm)

# 📝 Informe detallat
print("\nClassification report:\n", classification_report(yte, preds, target_names=["benigne","maliciós"]))

# 📈 Visualització de la pèrdua
import matplotlib.pyplot as plt

plt.figure()
plt.plot(range(10, epochs+1, 10), loss_hist, marker="o")
plt.xlabel("Època")
plt.ylabel("Train BCE")
plt.title("Evolució de la pèrdua (entrenament)")
plt.grid(True)
plt.show()
