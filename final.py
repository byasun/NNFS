import numpy as np
import nnfs
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data

nnfs.init()

# ===========================
# Camada Densa (Fully Connected)
# ===========================
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2. / n_inputs)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

# ===========================
# Ativação LeakyReLU
# ===========================
class Activation_LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.where(inputs > 0, inputs, self.alpha * inputs)

    def backward(self, dvalues):
        self.dinputs = np.where(self.inputs > 0, dvalues, self.alpha * dvalues)

# ===========================
# Ativação Softmax
# ===========================
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

# ===========================
# Função de Perda (Categorical Crossentropy)
# ===========================
class Loss_CategoricalCrossentropy:
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        return -np.log(correct_confidences)

    def calculate(self, output, y):
        return np.mean(self.forward(output, y))

    def backward(self, y_pred, y_true):
        samples = len(y_pred)
        if len(y_true.shape) == 1:
            y_true = np.eye(y_pred.shape[1])[y_true]
        self.dinputs = (y_pred - y_true) / samples
        return self.dinputs

# ===========================
# Otimizador Adam
# ===========================
class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iterations = 0

    def update_params(self, layer):
        if not hasattr(layer, 'm_w'):
            layer.m_w = np.zeros_like(layer.weights)
            layer.v_w = np.zeros_like(layer.weights)
            layer.m_b = np.zeros_like(layer.biases)
            layer.v_b = np.zeros_like(layer.biases)

        layer.m_w = self.beta1 * layer.m_w + (1 - self.beta1) * layer.dweights
        layer.m_b = self.beta1 * layer.m_b + (1 - self.beta1) * layer.dbiases

        layer.v_w = self.beta2 * layer.v_w + (1 - self.beta2) * (layer.dweights ** 2)
        layer.v_b = self.beta2 * layer.v_b + (1 - self.beta2) * (layer.dbiases ** 2)

        m_w_corr = layer.m_w / (1 - self.beta1 ** (self.iterations + 1))
        m_b_corr = layer.m_b / (1 - self.beta1 ** (self.iterations + 1))
        v_w_corr = layer.v_w / (1 - self.beta2 ** (self.iterations + 1))
        v_b_corr = layer.v_b / (1 - self.beta2 ** (self.iterations + 1))

        layer.weights -= self.learning_rate * m_w_corr / (np.sqrt(v_w_corr) + self.epsilon)
        layer.biases -= self.learning_rate * m_b_corr / (np.sqrt(v_b_corr) + self.epsilon)

        self.iterations += 1

# ===========================
# Gerando os dados
# ===========================
X, y = spiral_data(samples=100, classes=3)

# Criando a rede neural
dense1 = Layer_Dense(2, 64)
activation1 = Activation_LeakyReLU()

dense2 = Layer_Dense(64, 32)
activation2 = Activation_LeakyReLU()

dense3 = Layer_Dense(32, 3)
activation3 = Activation_Softmax()

loss_function = Loss_CategoricalCrossentropy()
optimizer = Optimizer_Adam(learning_rate=0.001)

epochs = 10000

# Listas para armazenar os valores de Loss e Accuracy
losses = []
accuracies = []

# ===========================
# Loop de Treinamento
# ===========================
for epoch in range(epochs):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)

    loss = loss_function.calculate(activation3.output, y)
    predictions = np.argmax(activation3.output, axis=1)
    accuracy = np.mean(predictions == y)

    losses.append(loss)
    accuracies.append(accuracy)

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')
        plt.imshow(dense1.weights, cmap='coolwarm', aspect='auto')
        plt.colorbar(label='Pesos')
        plt.title(f'Pesos da Primeira Camada - Epoch {epoch}')
        plt.xlabel('Neurônios')
        plt.ylabel('Entradas')
        plt.show()

    loss_function.backward(activation3.output, y)
    activation3.backward(loss_function.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)

# ===========================
# Gráficos da Loss e Accuracy
# ===========================
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses, label='Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(accuracies, label='Accuracy', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# ===========================
# Fronteira de Decisão
# ===========================
x_min, x_max = X[:, 0].min()-0.5, X[:, 0].max()+0.5
y_min, y_max = X[:, 1].min()-0.5, X[:, 1].max()+0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid = np.c_[xx.ravel(), yy.ravel()]

dense1.forward(grid)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)

Z = np.argmax(activation3.output, axis=1).reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.jet)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.jet, edgecolors='k')
plt.show()
