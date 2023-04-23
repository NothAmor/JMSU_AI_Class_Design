from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
import numpy as np
import os

number_file_count = []
max_num = 0

for i in range(10):
    for filename in os.listdir('CNN_LeNet_test'):
        if filename[0] == str(i):
            num_str = filename.split('_')[-1].split('.')[0]
            num = int(num_str)
            max_num = max(max_num, num)

    number_file_count.append(max_num)
    max_num = 0

# 加载训练数据
X_train, y_train = [], []
for i in range(10):
    for j in range(number_file_count[i]):
        img_path = f'CNN_LeNet_test/{i}_{j}.bmp'
        img = np.array(Image.open(img_path).convert('L')) / 255.0
        X_train.append(img.flatten())
        y_train.append(i)

# 构建模型并进行训练
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
model.fit(X_train, y_train)

# 对测试数据进行预测
test_dir = 'CNN_LeNet_test/test_image'
y_true, y_pred = [], []
for filename in os.listdir(test_dir):
    if filename.endswith('.bmp'):
        img_path = os.path.join(test_dir, filename)
        img = np.array(Image.open(img_path).convert('L')) / 255.0
        X_test = img.flatten().reshape(1, -1)
        y_pred.append(model.predict(X_test)[0])
        y_true.append(int(filename[0]))

# 计算识别准确率
accuracy = accuracy_score(y_true, y_pred)
print(f'识别准确率为：{accuracy:.2f}')
