import numpy as np
from matplotlib import pyplot as plt

np.random.seed(123)  # reproducibility
# import theano
from keras.models import Sequential  # 这是一个简单的线性神经网络层，它非常适合用在我们正在构建的前馈CNN中
from keras.layers import Dense, Dropout, Activation, Flatten  # 导入核心层
from keras.layers import Conv2D, MaxPooling2D  # 导入CNN层，这些卷积层可以帮助我们有效处理图像数据
from keras.utils import np_utils  # 加载一些工具模块，这些模块可以帮助我们进行数据转换

# 从MNIST加载图片数据
from keras.datasets import mnist

# Load pre-shuffled MNIST data into train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 查看数据集的维度
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# 绘制第一个样本
# plt.imshow(x_train[0])
# plt.show()
# 模型输入的二维向量
num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')
# 把数据类型转换为float32，并且把数据值归一化到[0,1]的区间
x_train /= 255
x_test /= 255

# y_train和y_test数据集并没有被拆分为10个不同的分类标签，而是一个包含有类型值的一维数组
# convert 1D class arrays to 10D class matrics
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test)  # 两者等效
num_classes = y_test.shape[1]


# print(y_train[0], y_test[0], y_test.shape[1])

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(200, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #机器学习模型的学习率,最朴素的accuracy
    return model

#build model
model = baseline_model()
#fit model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200, verbose=2)
#verbose = 2 为每个epoch输出一行记录
#final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print('Baseline Error: %.2f%%' % (100 - scores[1] * 100))