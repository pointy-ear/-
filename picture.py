import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("best_model.pkl", "rb") as f:
    params = pickle.load(f)

W1 = params['W1']  # shape: (3072, hidden_size)
hidden_size = W1.shape[1]

plt.figure(figsize=(15, 10))
for i in range(min(36, hidden_size)):
    ax = plt.subplot(6, 6, i + 1)
    img = W1[:, i].reshape(3, 32, 32).transpose(1, 2, 0)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))  # normalize to [0, 1]
    plt.imshow(img)
    plt.axis('off')
plt.tight_layout()
plt.savefig("weights_visualization.png")
