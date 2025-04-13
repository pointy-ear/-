手工实现的神经网络分类器（无 PyTorch/TF），用于在 CIFAR-10 数据集上进行训练和分类任务。

 训练模型
python train.py

 测试模型
python test.py

可视化权重
python picture.py

训练过程中会自动保存验证集最优的模型到 best_model.pkl

训练和验证的 loss/accuracy 曲线将保存为 loss_curve.png 和 val_accuracy_curve.png -
