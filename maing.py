import math
import os
import random

import cv2


class Image():
    def __init__(self, img, label=None):
        self.pixel_list = img
        self.label = label


class Neuron():
    def __init__(self, weights=None, bias=None):
        if weights == None:
            weights = []
        if bias == None:
            bias = 0
        self.weights = weights
        self.bias = bias

    def create_neuron(self, weights_number, weight_random_range=5, bias_random_range=10):
        # self.bias = self.randomize(bias_random_range)
        self.bias = 0
        for i in range(weights_number):
            self.weights.append(self.randomize(weight_random_range))

    def randomize(self, random_range):
        return random.random()-random.random()+random.randint(-1*random_range, random_range)


class Layer():
    def __init__(self, function, neurons_list=None):
        if neurons_list == None:
            neurons_list = []
        self.neurons_list = neurons_list
        self.function = function

    def add_neuron(self, neuron):
        self.neurons_list.append(neuron)


class NeuralNetwork():
    def __init__(self, struct):
        self.structure = struct
        self.neurons = []

    def create_network(self):
        # this creates a new layers list
        # where the first item is how many weights each neuron has
        # and the second one is how many neurons are there per layer
        neurons_structure = [i[0] for i in self.structure]
        layer_structure = [i[1] for i in self.structure]
        layer_structure.pop(0)

        input_layer = neurons_structure[0]
        layers_2 = []
        f = True
        for neurons_number in neurons_structure[1:]:
            if f:
                f = False
                layers_2.append((input_layer, neurons_number))
                previous_layer_neuron_number = neurons_number
                continue
            layers_2.append((previous_layer_neuron_number, neurons_number))
            previous_layer_neuron_number = neurons_number

        # this creates each neuron with the respective number of weights, then add this neurons to the neurons list
        for layer, function in zip(layers_2, layer_structure):
            lay = Layer(function)
            for i in range(layer[1]):
                neu = Neuron()
                neu.create_neuron(layer[0])
                lay.add_neuron(neu)
            self.neurons.append(lay)
        print('Network created with sucess!')

    def save_network(self, folder_name='config', file_prefix='config_'):
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        os.chdir(folder_name)

        for i in range(100):
            f = f'{file_prefix}{i}'
            if not os.path.exists(f):
                break

        with open(f, 'w') as file:
            text = f'{self.structure}'
            for layers in self.neurons:
                for neuron in layers.neurons_list:
                    weights = str(neuron.weights).translate(
                        {ord(i): None for i in ',[]'})
                    text += '\n'+str(neuron.bias)+'b'+weights
                text += '\nlayer'
            file.write(text)
        os.chdir('..')
        print('Network saved with sucess!')

    def load_network(self, file_name):
        os.chdir('config')
        with open(file_name, 'r') as file:
            lines = file.readlines()

        # this code check if the structure declared in the file, is the same from the Network
        struc = lines.pop(0)
        struc = struc.replace(',', '').replace('[', '').replace(
            ']', '').replace('(', '').replace(')', '').replace("'", '').replace('\n', '').split(' ')
        struc = [int(i) if i.isdigit() else i for i in struc]
        struc = [(struc[i], struc[i+1])
                 for i in [j for j in range(len(struc)) if j % 2 == 0]]
        if struc != self.structure:
            raise Exception(
                "Structure of loaded network don't match the structure declared!")

        layer_structure = [i[1] for i in self.structure]
        layer_structure.pop(0)

        layers_numbers = []

        new_lines = []
        while len(lines) != 0:
            line = lines.pop(0)
            if line == 'layer\n' or line == 'layer':
                layers_numbers.append(new_lines)
                new_lines = []
                continue
            new_lines.append(line)

        for layer, function in zip(layers_numbers, layer_structure):
            l = Layer(function)
            for neuron in layer:
                neuron_values = neuron.split('b')
                bias = float(neuron_values.pop(0))
                weights = neuron_values.pop(0)
                weights = weights.replace('\n', '').split(' ')

                neu = Neuron(bias=bias, weights=[
                    float(w) for w in weights])
                l.add_neuron(neu)
            self.neurons.append(l)
        os.chdir('..')
        print('Network loaded with sucess!')

    def next_layer_value(self, matrix, vector):
        # matrix -> layer
        # vector -> vector
        function = matrix.function

        layer_output = []
        for neuron in matrix.neurons_list:
            neuron_value = 0
            for num, weight in enumerate(neuron.weights):
                neuron_value += weight*vector[num]
            neuron_value += neuron.bias
            layer_output.append(neuron_value)

        if function == 'softmax':
            softmax_sum = 0
            for value in layer_output:
                softmax_sum += math.e**value

        for num, value in enumerate(layer_output):
            match function:
                case 'sigmoid':
                    value = 1/(1+(math.e ** (-1*value)))
                case 'relu':
                    value = max(0, value)
                case 'tanh':
                    value = math.tanh(value)
                case 'softmax':
                    value = (math.e ** value)/softmax_sum
                case _:
                    raise NotImplementedError(
                        f'The following function defined for the layer is not implemented: {function}')

            layer_output[num] = value
        return layer_output

    def get_weight(self, prev_layer_neuron, this_layer_neuron, target_value, function):
        pass

    def train_network(self, epochs, learning_rate=0.5, path='trainingSet'):

        # backpropagation
        os.chdir(path)
        labels = os.listdir()

        images = []
        for lab in labels:
            os.chdir(lab)
            image_list = os.listdir()
            a = 0
            for img_path in image_list:
                a += 1
                if a == 100:
                    break

                image = cv2.imread(img_path)
                image = image.tolist()
                for i in range(28):
                    for j in range(28):
                        gray_value = 0
                        for m in range(3):
                            gray_value += image[i][j][m]
                        gray_value = gray_value/3
                        image[i][j] = round(gray_value/255, 10)
                images.append(Image([i for j in image for i in j], lab))
            os.chdir('..')
        os.chdir('..')

        raise(NotImplementedError)
        
def main():
    NeuralNet = NeuralNetwork(
        [(784, 'input'), (30, 'sigmoid'), (30, 'sigmoid'), (10, 'softmax')])
    # NeuralNet.create_network()
    # NeuralNet.save_network()

if __name__ == '__main__':
    main()
