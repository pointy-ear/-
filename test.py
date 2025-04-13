# 文件：test.py
import numpy as np
import pickle
from models.neural_net import NeuralNetwork
from train import download_and_extract_cifar10, load_batch, preprocess
from sklearn.metrics import accuracy_score
import os

# 加载测试集数据
def load_test_data():
    path = download_and_extract_cifar10()
    X_test, y_test = load_batch(os.path.join(path, "test_batch"))
    X_test = preprocess(np.array(X_test))
    y_test = np.array(y_test)
    return X_test, y_test

# 测试函数
def test():
    input_size = 32 * 32 * 3
    hidden_size = 128  # 确保与训练时一致
    output_size = 10
    reg_lambda = 0.001

    model = NeuralNetwork(input_size, hidden_size, output_size, activation='relu', reg_lambda=reg_lambda)

    # 加载保存的参数
    with open("best_model.pkl", "rb") as f:
        best_params = pickle.load(f)
        model.params = best_params

    # 加载测试数据
    X_test, y_test = load_test_data()

    # 前向传播预测
    _, probs = model.compute_loss(X_test, y_test)
    y_pred = np.argmax(probs, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    test()
