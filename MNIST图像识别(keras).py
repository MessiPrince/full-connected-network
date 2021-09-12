import numpy as np
import scipy.special


class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes
        self.rate = learning_rate
        self.w_ih = np.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.w_ho = np.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))
        self.activate = lambda x: scipy.special.expit(x)  # 激活函数设置为sigmoid函数

    def train(self, inputs_list, targets_list):
        # 转换为二维数组
        inputs = np.array(inputs_list, ndmin=2).T
        # 目标值
        targets = np.array(targets_list, ndmin=2).T
        # 隐藏层输入
        hidden_in = np.dot(self.w_ih, inputs)
        # 隐藏层输出
        hidden_out = self.activate(hidden_in)
        # 输出层输入
        outputs_in = np.dot(self.w_ho, hidden_out)
        # 输出层输出
        outputs_out = outputs_in
        # 反向传播计算网络误差
        output_error = targets - outputs_out  # Loss，实际上应该用的是均方差函数（2放到rate里面）
        hidden_error = np.dot(self.w_ho.T, output_error)
        # 输出层和隐藏层之间的权重更新
        self.w_ho += self.rate * np.dot(output_error, np.transpose(hidden_out))
        # 输入层和隐藏层之间的权重更新
        self.w_ih += self.rate * np.dot((hidden_error * hidden_out * (1.0 - hidden_out)), np.transpose(inputs))

    def query(self, inputs_list):  # 接收输入，返回输出
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_in = np.dot(self.w_ih, inputs)
        hidden_out = self.activate(hidden_in)
        outputs_in = np.dot(self.w_ho, hidden_out)
        outputs_out = outputs_in
        return outputs_out


# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning rate
learning_rate = 0.01  # 不能太大，大于0.2好像就不行了

# create instance
network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# load csv file
# 打开并读取数据集
# 从MNIST加载图片数据
from keras.datasets import mnist
from keras.utils import np_utils  # 加载一些工具模块，这些模块可以帮助我们进行数据转换

# Load pre-shuffled MNIST data into train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')  # 60000行，784列数组
x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')
x_train /= 255
x_test /= 255
# convert 1D class arrays to 10D class matrics: one-hot->10, 60000 * 10
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# repeat = 4
# for i in range(repeat):
for i in range(x_train.shape[0]):
    network.train(x_train[i], y_train[i])

result = []
for i in range(x_test.shape[0]):
    correct_label = np.argmax(y_test[i])
    print('correct:', correct_label)
    outputs = network.query(x_test[i])
    label = np.argmax(outputs)  # index
    print('test:', label)
    if label == correct_label:
        result.append(1)
    else:
        result.append(0)
        pass
    pass
print(result)

result_array = np.asfarray(result)
print('correct rate:', result_array.sum() * 1.0 / result_array.size)
print(result_array.size)
