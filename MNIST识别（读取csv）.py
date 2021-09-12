import numpy as np
import matplotlib.pyplot
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
learning_rate = 0.01  # 不能太大

# create instance
network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# load csv file
# 打开并读取数据集
training_data = open('./number/mnist_dataset/mnist_train.csv', 'r')
training_list = training_data.readlines()  # 一行的字符串（每一项）
training_data.close()

# 2.将用逗号分隔的数字列表转换成合适的数组，先拆分，后转换为数组，最后绘制数组。
# split函数告诉一个函数按照那个符号进行拆分，并将拆分后的结果放在all_values当中
# all_values[1:]指的是除了列表中第一个元素以外的值，numpy.asfarray表示将文本字符串转为实数，并创建这些数字的数组。对多维数组做数值处理
# .reshape表示每28个元素折返一次，最终形成了28*28的矩阵
# imshow显示绘制image_array，选择灰度调色板cmap=’Greys’
# 准备MNIST训练数据
# （1）0-255缩放至0.01-1.0，避免设置为0而导致权重更新失败
# repeat = 4
# for i in range(repeat):
for record in training_list:
    all_values = record.split(',')
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01  # list -> array
    targets = np.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    network.train(inputs, targets)

# test
# (1)
test_data_file = open('./number/mnist_dataset/mnist_test.csv', 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
# all_values = test_data_list[0].split(',')
# print(all_values[0])
result = []
for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    print('correct:', correct_label)
    outputs = network.query((np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
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
# len只计算第一层，size计算最底层
