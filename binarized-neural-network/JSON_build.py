import json
from tensorflow.python import pywrap_tensorflow
import os
import numpy as np
import re
import sys

# get numbers of layers
def get_number_of_layers(var_to_shape_map):
    numbers = []
    for key in var_to_shape_map:
        m = re.search('_layer(.+?)', key)
        if m:
            found = m.group(1)
            numbers.append(found)

    number_of_layers = max(numbers)
    return number_of_layers


# build layer for binaryDense
def build_layer_linarization(var_to_shape_map, number_of_layers):
    layer_linarization = {}

    for x in range(1, int(number_of_layers) + 1):
        string1 = "layer" + str(x) + "/weight_binary"
        string2 = "layer" + str(x) + "/bias"
        string3 = "layer" + str(x) + "/number_of_neurons"
        weight_dense = []
        bias_dense = []
        number_of_neurons = 0
        name = ""
        for key in var_to_shape_map:
            if "batchNormalization" not in key and "Adam" not in key:
                if string1 in key:
                    weight_dense.append(reader.get_tensor(key).tolist())
                    name = key
                if string2 in key:
                    bias_dense.append((reader.get_tensor(key)).tolist())
                if string3 in key:
                    number_of_neurons = reader.get_tensor(key).tolist()

        layer_linarization[x] = {
        "nameLayer": name.split("_layer")[0],
        "weight": weight_dense,
        "bias": bias_dense,
        "number_of_neurons": number_of_neurons
        }
    return layer_linarization

# build layer for batchNormalization
def build_layer_batchnormalization(var_to_shape_map, number_of_layers):
    layer_batchnormalization = {}

    for x in range(1, int(number_of_layers)+1):
        string1 = "layer" + str(x) +"/beta"
        string2 = "layer" + str(x) +"/gamma"
        string3 = "layer" + str(x) +"/moving_mean"
        string4 = "layer" + str(x) +"/moving_variance"
        alpha = []
        gamma = []
        miu = []
        sigma = []
        name = ""

        for key in var_to_shape_map:
            if "batchNormalization" in key and "Adam" not in key:
                if string1 in key:
                    name = key
                    gamma.append(reader.get_tensor(key).tolist())
                if string2 in key:
                    alpha.append(reader.get_tensor(key).tolist())
                if string3 in key:
                    miu.append(reader.get_tensor(key).tolist())
                if string4 in key:
                    sigma.append(reader.get_tensor(key).tolist())

        layer_batchnormalization[x] = {
            "nameLayer": name.split("_layer")[0],
            "alpha": alpha,
            "gamma": gamma,
            "miu": miu,
            "sigma": sigma
        }

    return layer_batchnormalization

#build network
def build_network(layer_linarization,layer_batchnormalization, number_of_layers, pathName):
    for x in range(1, int(number_of_layers) + 1):
        network = {
            "nameNetwork": pathName.split("/models/",1)[1],
            "number_of_layers": number_of_layers,
            "layers_linarization": layer_linarization,
            "layers_batchNormalization": layer_batchnormalization
            }
    return network

if __name__== "__main__":

    # select the path for your model
    pathName = "/home/razvan/PycharmProjects/BNN_version2/binarized-neural-network/models/1581520202_binary_sbn_cifar10"

    # read data from model
    checkpoint_path = os.path.join(pathName, "model.ckpt")
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()

    # build the network for your model
    number_of_layers = get_number_of_layers(var_to_shape_map)
    layer_linarization = build_layer_linarization(var_to_shape_map, number_of_layers)
    layer_batchnormalization = build_layer_batchnormalization(var_to_shape_map, number_of_layers)
    network = build_network(layer_linarization, layer_batchnormalization, number_of_layers, pathName)

    # build the JSON file
    json_network = json.dumps(network, indent=6)
    nameFile = pathName.split("/models/",1)[1],
    jsonFilePath = "/home/razvan/PycharmProjects/BNN_version2/binarized-neural-network/" + pathName.split("/models/",1)[1] + ".json"
    f= open(jsonFilePath, "w+")
    f.write(json_network)
    f.close()

