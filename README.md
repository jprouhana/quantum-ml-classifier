# quantum-ml-classifier

compares variational quantum classifier + quantum kernel SVM against classical SVM on toy datasets. built with qiskit ml.

classical svm still wins on small datasets but the quantum pipeline is interesting.

## setup

```
pip install -r requirements.txt
```

## usage

```python
from src.quantum_classifier import train_vqc
from src.data_utils import load_moons_dataset

X_train, X_test, y_train, y_test = load_moons_dataset()
model, obj_values = train_vqc(X_train, y_train)
```

full comparison in `notebooks/quantum_classification.ipynb`.
