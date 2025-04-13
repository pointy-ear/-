# 文件：train.py
import numpy as np
import pickle
import matplotlib.pyplot as plt
from models.neural_net import NeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import tarfile
import urllib.request

# 下载和解压 CIFAR-10 数据集（官方 Python 版本）
def download_and_extract_cifar10(download_dir="data"):
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = os.path.join(download_dir, "cifar-10-python.tar.gz")
    extracted_dir = os.path.join(download_dir, "cifar-10-batches-py")

    if not os.path.exists(extracted_dir):
        os.makedirs(download_dir, exist_ok=True)
        print("Downloading CIFAR-10...")
        urllib.request.urlretrieve(url, filename)
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(path=download_dir)
    return extracted_dir

# 加载单个 batch 文件
def load_batch(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
        X = dict[b'data']
        y = dict[b'labels']
        return X, y

# 加载完整 CIFAR-10 数据集
def load_cifar10_numpy():
    path = download_and_extract_cifar10()
    X_list, y_list = [], []
    for i in range(1, 6):
        X, y = load_batch(os.path.join(path, f"data_batch_{i}"))
        X_list.append(X)
        y_list.extend(y)
    X_train = np.concatenate(X_list)
    y_train = np.array(y_list)

    X_test, y_test = load_batch(os.path.join(path, "test_batch"))
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test

# 预处理函数
def preprocess(X):
    X = X.astype(np.float32) / 255.0
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-7
    return (X - mean) / std

def load_data():
    X_train, y_train, X_test, y_test = load_cifar10_numpy()
    X_train = preprocess(X_train)
    X_test = preprocess(X_test)
    return train_test_split(X_train, y_train, test_size=0.2, random_state=42), (X_test, y_test)

# 训练函数
def train():
    (X_train, X_val, y_train, y_val), (X_test, y_test) = load_data()
    input_size = 32 * 32 * 3
    hidden_size = 256
    output_size = 10
    learning_rate = 0.15
    reg_lambda = 0.0001
    epochs = 30
    decay = 0.95
    batch_size = 128

    model = NeuralNetwork(input_size, hidden_size, output_size, activation='relu', reg_lambda=reg_lambda)

    train_losses, val_losses, val_accs = [], [], []
    best_acc = 0
    best_params = None

    num_train = X_train.shape[0]
    num_batches = num_train // batch_size

    for epoch in range(epochs):
        permutation = np.random.permutation(num_train)
        X_train = X_train[permutation]
        y_train = y_train[permutation]

        epoch_loss = 0
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            loss, probs = model.compute_loss(X_batch, y_batch)
            grads = model.backward(X_batch, y_batch, probs)
            model.update_params(grads, learning_rate)
            epoch_loss += loss

        epoch_loss /= num_batches

        val_loss, val_probs = model.compute_loss(X_val, y_val)
        val_preds = np.argmax(val_probs, axis=1)
        val_acc = accuracy_score(y_val, val_preds)

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}: train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_params = {k: v.copy() for k, v in model.params.items()}

        learning_rate *= decay

    # 保存最优模型
    with open("best_model.pkl", "wb") as f:
        pickle.dump(best_params, f)

    # 可视化
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.clf()

    plt.plot(val_accs, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("val_accuracy_curve.png")

if __name__ == '__main__':
    train()
