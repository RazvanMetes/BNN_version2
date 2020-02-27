from tensorflow.python import pywrap_tensorflow
import os
import numpy as np

checkpoint_path = os.path.join("/home/razvan/PycharmProjects/BNN_version2/binarized-neural-network/models/1581520202_binary_sbn_cifar10","model.ckpt")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()

f= open("guru99.txt","w+")

np.set_printoptions(threshold=np.inf)

for key in var_to_shape_map:
    if "" in key:
        print("tensor_name: ", key)
        #print(reader.get_tensor(key))
        f.write("tensor_name" + key+ "\n")
        f.write(str(reader.get_tensor(key)) + "\n")
