# Quantum Experiments Open

Aquest repositori forma part del projecte **Quantum Experiments Open**, una iniciativa oberta -d'en Jordi Garcia Castillón (CibraLAB)- per explorar l’ús de la **computació quàntica aplicada a la intel·ligència artificial i la ciberseguretat**.

## 📌 Descripció de l’experiment

El codi d’aquest projecte implementa un **classificador binari quàntic** utilitzant [PennyLane](https://pennylane.ai/) i [Amazon Braket](https://aws.amazon.com/braket/).  

L’objectiu és distingir entre dues classes de dades (en aquest exemple: `benigne` i `maliciós`) a partir d’un conjunt d’entrenament i validació.

### Punts clau
- **Arquitectura híbrida quàntica-clàssica**:  

  El model combina circuits quàntics parametritzats (variational quantum circuits, VQCs) amb optimització clàssica mitjançant l’optimitzador `Adam`.
  
- **Dispositiu SV1 d’Amazon Braket**:  

  L’execució principal està configurada per utilitzar el simulador **SV1** de Braket (`braket.aws.qubit`), que és un simulador de vectors d’estat amb fins a 34 qubits.  

  També es pot executar en local (`default.qubit` o `lightning.qubit`) per fer proves més ràpides.

- **Funció de pèrdua**:  

  Es fa servir la **Binary Cross-Entropy (BCE)** per mesurar la qualitat del classificador.

- **Entrenament**:  
  - Entrena el circuit amb batches aleatoris.  
  - Imprimeix la pèrdua cada 10 èpoques.  
  - Desa un històric de la pèrdua per a visualització.

- **Avaluació**:
  Al final de l’entrenament calcula:
  - Accuracy del test  
  - Matriu de confusió  
  - Informe de classificació (precision, recall i F1-score)  
  - Gràfic de l’evolució de la pèrdua

## 📂 Estructura del codi

1. **Inicialització de paràmetres**
    
   ```python
   params = 0.01 * np.random.randn(layers, n_qubits)

   ## Inicialització dels paràmetres

Els paràmetres del circuit quàntic es inicialitzen amb valors aleatoris petits.

2. **Definició de la pèrdua BCE**

   ```python
   def bce_loss(params, Xb, yb):

   ## Definició de pèrdua BCE

3. **Optimització**

   ```python
   opt = qml.AdamOptimizer(stepsize=0.05)
   for epoch in range(epochs):

   ## Optimització

4. **Avaluació del model**
   
Es comparen les prediccions amb les etiquetes reals i es mostren mètriques.

5. **Visualització**
   
Es dibuixa l’evolució de la pèrdua al llarg de l’entrenament.

---

## 🚀 Execució

1. **Entorn recomanat**
   
Un notebook a **Amazon Braket** (exemple: instància `ml.t3.medium`).

**Python 3.9 o superior** amb les dependències següents:

- `pennylane`  
- `amazon-braket-sdk`  
- `scikit-learn`  
- `matplotlib`  

2. **Dispositiu per defecte**
   
El codi utilitza **SV1 de Braket**.

```python
dev = qml.device(
    "braket.aws.qubit",
    device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
    wires=4,
    shots=None
)

```
3. **Execució local (opcional)**

Per iterar més ràpidament:

```python
dev = qml.device("default.qubit", wires=4, shots=200)
# o millor si està disponible:
# dev = qml.device("lightning.qubit", wires=4, shots=200)

```

## 📊 Resultats esperats

- L’accuracy de test pot variar segons les dades i la inicialització, típicament entre **0.6 i 0.8** en conjunts petits o sintètics.  
- El gràfic mostra la disminució de la **pèrdua BCE** amb les èpoques.  
- Les **mètriques de classificació** donen una visió clara de com es comporta el classificador per cada classe.

## ⚖️ Llicència

Aquest projecte és open source i es publica sota la llicència MIT
.

✍️ Quantum Experiments Open és una iniciativa comunitària "AI-Powered" per aprendre, compartir i explorar els límits de la computació quàntica aplicada a problemes reals, especialment, de la seguretat de la IA.

---

## 🔀 Variants de l'algorisme

Aquest repositori -d'inici- inclou **dues opcions del classificador VQC (Variational Quantum Classifier):**

- **`vqc_classifier.py`**
   
  Versió **completa** de l’algorisme.  

  Proporciona resultats més fidels i detallats, però és més **lenta** d’executar, especialment quan s’utilitza el simulador SV1 al núvol.

- **`vqc_classifier_fast.py`**  

  Versió **lleugera i ràpida**, pensada per a **proves i testejos inicials**.  

  Redueix el nombre d’èpoques, la mida dels batches i la quantitat de dades avaluades, permetent obtenir resultats en segons amb un simulador local.

✅ Ambdues variants es poden executar i adaptar lliurement segons les necessitats de cada usuari.
