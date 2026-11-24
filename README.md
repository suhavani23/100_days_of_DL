# 🧠 100 Days of Deep Learning

Welcome to **100 Days of Deep Learning**! This repository documents a structured, hands-on journey through Deep Learning fundamentals, mathematical derivations, custom implementations from scratch, and real-world neural network architectures built with modern frameworks.

---

## 📌 Repository Roadmap

The repository is organized progressively, starting with the simplest computational unit—the Perceptron—and expanding into multi-layer networks, backpropagation calculus from scratch, gradient dynamics, and full-scale Artificial Neural Network (ANN) projects.

```
100_days_of_DL/
│
├── 📂 Perceptron/                            # Foundations of Perceptrons & Single-Layer Units
│   ├── perceptron-demo.ipynb                 # Scikit-Learn Perceptron demo & decision boundary visualization
│   ├── perceptron-trick.ipynb                # Geometric Perceptron Trick implemented from scratch
│   ├── hinge-loss-perceptron.ipynb           # Perceptron loss function (Hinge loss) & gradient descent
│   ├── Problem_with_perceptron.ipynb         # Non-linear separability (XOR problem) & limitations
│   └── placement.csv                         # 2D classification dataset (CGPA, IQ, Placement)
│
├── 📂 models/                                # Applied Deep Learning & Neural Network Projects
│   ├── Churn_rate_prediction.ipynb           # Customer churn binary classification using Keras Sequential ANN
│   ├── Grad_pred_ANN.ipynb                   # Graduate university admission regression using ANN
│   ├── mnist_classification.ipynb            # Multi-class handwritten digit recognition on MNIST dataset
│   ├── Admission_Predict_Ver1.1.csv          # University admission dataset
│   └── Churn_Modelling.csv                   # Bank customer churn dataset
│
├── 📓 MLP_Notation.ipynb                     # Mathematical notations, dimensions & forward propagation theory
├── 📓 backpropagation_classification.ipynb   # Complete L-layer backpropagation for classification from scratch
├── 📓 backpropagation_regression.ipynb       # Complete L-layer backpropagation for regression from scratch
├── 📓 vanishing_gradient.ipynb               # In-depth analysis & visualization of vanishing/exploding gradients
├── 📄 .gitignore                             # Git ignore rules for checkpoints & virtual environments
└── 📄 README.md                              # Repository documentation & guide
```

---

## 🔬 Topics & Modules Covered

### 1. Perceptron & Foundations of Neural Units (`/Perceptron`)
* **Perceptron Demo (`perceptron-demo.ipynb`)**:
  * Using `sklearn.linear_model.Perceptron` on campus placement data.
  * Plotting linear decision boundaries separating classes based on CGPA and test scores.
* **The Perceptron Trick (`perceptron-trick.ipynb`)**:
  * Pure Python/NumPy implementation of the perceptron update rule without black-box libraries.
  * Geometric intuition of shifting and rotating the decision boundary hyperplane using:
    $$W_{new} = W_{old} + \eta \cdot (y_i - \hat{y}_i) \cdot X_i$$
    $$b_{new} = b_{old} + \eta \cdot (y_i - \hat{y}_i)$$
* **Hinge Loss & Loss-based Formulation (`hinge-loss-perceptron.ipynb`)**:
  * Why step functions are non-differentiable and unsuitable for gradient-based optimization.
  * Formulating the Perceptron loss function:
    $$L(w, b) = -\sum_{i \in \mathcal{M}} y_i (w \cdot x_i + b)$$
  * Minimizing the loss via Batch and Stochastic Gradient Descent.
* **Limitations of Single-Layer Perceptrons (`Problem_with_perceptron.ipynb`)**:
  * Linear separability constraints: why single-layer perceptrons fail on non-linear datasets (e.g., XOR gate, concentric circles).
  * The necessity of Multi-Layer Perceptrons (MLP) and non-linear activation functions.

---

### 2. Multi-Layer Perceptron (MLP) Architecture (`MLP_Notation.ipynb`)
* **Standard Mathematical Notation**:
  * Layer indexing: $l \in \{1, 2, \dots, L\}$.
  * Weight matrices $W^{[l]}$ and bias vectors $b^{[l]}$.
  * Pre-activation linear combination: $Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$.
  * Post-activation non-linear output: $A^{[l]} = \sigma(Z^{[l]})$.
* **Matrix Dimension Conventions**:
  * Rigorous dimensional consistency checks for batch vectorized computations:
    $$\text{dim}(W^{[l]}) = (n^{[l]}, n^{[l-1]}), \quad \text{dim}(b^{[l]}) = (n^{[l]}, 1)$$
* **Architectural Trade-offs**:
  * Network depth vs. width: capacity, representational power, and the Universal Approximation Theorem.

---

### 3. Backpropagation from Scratch
Full manual implementation of forward propagation, loss computation, and backpropagation via the chain rule using only **NumPy**:
* **Binary Classification (`backpropagation_classification.ipynb`)**:
  * Forward pass with arbitrary layer dimensions.
  * Binary Cross-Entropy (Log Loss) computation:
    $$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(a^{[L](i)}) + (1 - y^{(i)}) \log(1 - a^{[L](i)}) \right]$$
  * Backward pass calculating gradients $dW^{[l]}$, $db^{[l]}$, and $dA^{[l-1]}$ layer-by-layer.
* **Regression (`backpropagation_regression.ipynb`)**:
  * Linear activation in the final layer for continuous value estimation.
  * Mean Squared Error (MSE) loss and derivative propagation through hidden layers.
  * Gradient descent parameter updates with variable learning rates.

---

### 4. Deep Learning Mechanics & Debugging (`vanishing_gradient.ipynb`)
* **Vanishing & Exploding Gradients**:
  * Understanding why deep neural networks fail to train when using saturating activation functions (Sigmoid, Tanh).
  * Mathematical derivation of how repeated multiplications of derivatives $< 1$ cause early layer gradients to shrink exponentially:
    $$\frac{\partial \mathcal{L}}{\partial W^{[1]}} = \frac{\partial \mathcal{L}}{\partial A^{[L]}} \cdot \prod_{k=2}^{L} \left( \frac{\partial A^{[k]}}{\partial Z^{[k]}} W^{[k]} \right) \cdot \frac{\partial Z^{[1]}}{\partial W^{[1]}}$$
  * Layer-by-layer weight tracking and gradient inspection across training epochs.
  * Solutions: Modern non-saturating activations (ReLU, Leaky ReLU) and proper weight initialization strategies (He, Xavier/Glorot).

---

### 5. Applied Projects & Framework Architectures (`/models`)
* **Bank Customer Churn Prediction (`Churn_rate_prediction.ipynb`)**:
  * **Dataset**: `Churn_Modelling.csv` (10,000 customer records).
  * **Preprocessing Pipeline**: Handling high-cardinality categorical features (One-Hot Encoding `Geography`, Label Encoding `Gender`), feature scaling with `StandardScaler`.
  * **Model Architecture**: Deep Artificial Neural Network built with `tensorflow.keras.Sequential` and `Dense` layers.
  * **Evaluation**: Accuracy, ROC-AUC, and loss curves analysis across epochs.
* **Graduate Admissions Prediction (`Grad_pred_ANN.ipynb`)**:
  * **Dataset**: `Admission_Predict_Ver1.1.csv`.
  * **Task**: Regression predicting continuous university admission probability from GRE score, TOEFL score, university rating, SOP, LOR, CGPA, and research experience.
  * **Pipeline**: Feature normalization with `MinMaxScaler`, multi-layer regression ANN, and performance tracking (MSE/MAE).
* **Handwritten Digit Classification (`mnist_classification.ipynb`)**:
  * **Dataset**: Classic MNIST handwritten digits (70,000 $28 \times 28$ grayscale images).
  * **Architecture**: Input `Flatten` layer followed by hidden `Dense` layers with ReLU activations, and output `Dense` layer with Softmax.
  * **Optimization**: `sparse_categorical_crossentropy` loss with Adam optimizer, validation monitoring, and interactive test prediction visualizations.

---

## 🛠️ Tech Stack & Tools

* **Core Language**: Python 3.10+
* **Deep Learning Frameworks**: TensorFlow 2.x, Keras
* **Scientific Computing & Data**: NumPy, Pandas, Scikit-Learn
* **Data Visualization**: Matplotlib, Seaborn
* **Interactive Environment**: Jupyter Notebooks / VS Code

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/suhavani23/100_days_of_DL.git
cd 100_days_of_DL
```

### 2. Set Up a Virtual Environment
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook
```

---

## 📈 Learning Highlights & Key Takeaways

1. **Math First, Code Second**: True mastery comes from deriving the matrix calculus before writing `model.fit()`.
2. **From Perceptron to MLP**: Single units create hyperplanes; networks of units create arbitrary continuous decision boundaries.
3. **The Computational Graph**: Backpropagation is reverse-mode automatic differentiation applied to a chain of composite functions.
4. **Architectural Discipline**: Feature scaling, appropriate activation functions, and gradient health checks are critical to training deep models effectively.

---

## 👤 Author

* **Suhavani** - [GitHub (@suhavani23)](https://github.com/suhavani23)

⭐ If you find this repository helpful in your own Deep Learning journey, feel free to star the repo!
