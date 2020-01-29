from tensorflow.python import pywrap_tensorflow
import os


checkpoint_path = os.path.join("/home/razvan/PycharmProjects/BNN_version2/binarized-neural-network/models/1580225710_binary_mnist","model.ckpt")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()

for key in var_to_shape_map:
    if "binarydense1" in key:
        print("tensor_name: ", key)
        print(reader.get_tensor(key))