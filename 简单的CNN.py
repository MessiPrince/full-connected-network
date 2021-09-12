import numpy as np
from keras.models import Sequential  # 这是一个简单的线性神经网络层，它非常适合用在我们正在构建的前馈CNN中
from keras.layers import Dense, Dropout, Activation, Flatten  # 导入核心层
from keras.layers import Conv2D, MaxPooling2D  # 导入CNN层，这些卷积层可以帮助我们有效处理图像数据
from keras.utils import np_utils  # 加载一些工具模块，这些模块可以帮助我们进行数据转换
from keras import backend as K

# K.image_data_format = 'channels_last'
# K.set_image_data_format('tf')

# 从MNIST加载图片数据
from keras.datasets import mnist

# 将数据reshape，CNN的输入是4维的张量（可看做多维的向量），第一维是样本规模，第二维是像素通道，第三维和第四维是长度和宽度。并将数值归一化和类别标签向量化。
#
# # load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# padding = 'same'
# reshape to be [samples][pixels][width][height]

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')


x_train = x_train / 255
x_test = x_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# 第一层是卷积层。该层有32个feature map,或者叫滤波器，作为模型的输入层，接受[pixels][width][height]大小的输入数据。feature map的大小是5*5，其输出接一个‘relu’激活函数。
# 下一层是pooling层，使用了MaxPooling，大小为2*2。
# 下一层是Dropout层，该层的作用相当于对参数进行正则化来防止模型过拟合。
# 接下来是全连接层，有128个神经元，激活函数采用‘relu’。
# 最后一层是输出层，有10个神经元，每个神经元对应一个类别，输出值表示样本属于该类别的概率大小。

def baseline_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten()) #扁平化
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# build model
model = baseline_model()
# fit model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200, verbose=2)
# final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print('Baseline Error: %.2f%%' % (100 - scores[1] * 100))
