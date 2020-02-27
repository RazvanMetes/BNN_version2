import json
from ProblemDefinition import BNNProblem

def prepareBNNProblem(problem_file_name):
    with open(problem_file_name) as f:
        data = json.load(f)

    layer_list =[]
    number_of_layers =  data['number_of_layers']
    for i in range(1, int(number_of_layers) + 1):
        x = '' + str(i) + ''
        weight = data['layers_linarization'][x]['weight']
        bias = data['layers_linarization'][x]['bias']
        number_of_neurons = data['layers_linarization'][x]['number_of_neurons']
        alpha = data['layers_batchNormalization'][x]['alpha']
        gamma = data['layers_batchNormalization'][x]['gamma']
        miu = data['layers_batchNormalization'][x]['miu']
        sigma = data['layers_batchNormalization'][x]['sigma']


        layer_list.append( BNNProblem(i, weight, bias, number_of_neurons, alpha, gamma, miu, sigma))

    return layer_list

if __name__ == "__main__":
    bnnp = prepareBNNProblem("../training/out/modelsJSON/1581432976_binary_mnist.json")
    for obj in bnnp:
        print(obj.layer, obj.weight, obj.bias, obj.number_of_neurons, obj.alpha, obj.gamma, obj.miu, obj.sigma, sep=' ')
