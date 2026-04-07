import numpy as np 
class SVM:
    def __init__(self, learning_rate=0.01, C=1.0, n_iters=1000):
        self.lr = learning_rate
        self.C = C
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
       
        y_ = np.where(y <= 0, -1, 1)

        limit = 1 / np.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, n_features)
        self.b = 0

        for epoch in range(self.n_iters):
            
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y_[indices]

        
            lr_t = self.lr / (1 + epoch * 0.001)

            for idx, x_i in enumerate(X_shuffled):
            
                condition = y_shuffled[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                
                if condition:
            
                    self.w -= lr_t * (2 * 1/self.n_iters * self.w)
                else:
                    
                    self.w -= lr_t * (2 * 1/self.n_iters * self.w - self.C * y_shuffled[idx] * x_i)
                    self.b -= lr_t * (-self.C * y_shuffled[idx])

    def predict(self, X):
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx)






import numpy as np
import pennylane as qml
from sklearn.svm import SVC

N_QUBITS = 4
SHOTS    = 1024

dev = qml.device("default.qubit", wires=N_QUBITS, shots=SHOTS)

def pauli_feature_map(x: np.ndarray, reps: int = 2):

    for _ in range(reps):
      
        for i in range(N_QUBITS):
            qml.Hadamard(wires=i)
        for i in range(N_QUBITS):
            qml.RZ(2.0 * x[i], wires=i)

       
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
            qml.RZ(2.0 * x[i] * x[i + 1], wires=i + 1)
            qml.CNOT(wires=[i, i + 1])

@qml.qnode(dev)
def kernel_circuit(x1: np.ndarray, x2: np.ndarray) -> float:

    pauli_feature_map(x1, reps=2)
    qml.adjoint(pauli_feature_map)(x2, reps=2)
    return qml.probs(wires=range(N_QUBITS))


def quantum_kernel(x1: np.ndarray, x2: np.ndarray) -> float:
 
    probs = kernel_circuit(x1, x2)
    return float(probs[0])          # index 0 → |00…0⟩ state



def build_kernel_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    
    return qml.kernels.kernel_matrix(A, B, quantum_kernel)



class QSVC:


    def __init__(self, C: float = 1.0, shots: int = SHOTS, reps: int = 2):
        self.C     = C
        self.shots = shots
        self.reps  = reps
        self._svc  = SVC(kernel="precomputed", C=C, probability=True, class_weight="balanced")
        self.X_train_: np.ndarray | None = None

    # ── fit ──────────────────────────────────────────────────────
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "QSVC":
      
        self.X_train_ = X_train
        print(f"  [QSVC] Building train kernel matrix ({len(X_train)}×{len(X_train)})…")
        K_train = build_kernel_matrix(X_train, X_train)
        self._svc.fit(K_train, y_train)
        print("  [QSVC] Training complete.")
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
       
        K_test = build_kernel_matrix(X_test, self.X_train_)
        return self._svc.predict(K_test)

   
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        K_test = build_kernel_matrix(X_test, self.X_train_)
        return self._svc.predict_proba(K_test)

    
    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        return float((self.predict(X_test) == y_test).mean())



def kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
  
    Y   = np.outer(y * 2 - 1, y * 2 - 1)   # map {0,1} → {-1,+1}
    num = np.sum(K * Y)
    den = np.linalg.norm(K, "fro") * np.linalg.norm(Y, "fro") + 1e-9
    return float(num / den)