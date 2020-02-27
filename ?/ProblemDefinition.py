class BNNProblem:
    def __init__(self, layer, weight, bias, number_of_neurons, alpha, gamma, miu, sigma):
        self.layer = layer
        self.weight = weight
        self.bias = bias
        self.number_of_neurons = number_of_neurons
        self.alpha = alpha
        self.gamma = gamma
        self.miu = miu
        self.sigma = sigma