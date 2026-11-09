import numpy as np

class Node(object):

    def __init__(self, weights=None, bias=None, pre_value=None, value=0):
        """
        self - Node object 
        weights - List of floats
        bias - Single float 
        pre_value - float 
        value - float
        """
        
        # Define weights attribute
        self.weights = weights

        # Define bias attribute
        self.bias = bias

        # Define pre_value attribute
        self.pre_value = pre_value

        # Define value attribute
        self.value = value



    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def get_value(self):
        return self.value

    def get_pre_value(self):
        return self.pre_value

    def set_value(self, value):
        self.value = value

    def set_pre_value(self, pre_value):
        self.pre_value = pre_value

    def update_pre_value(self, last_vals):
        """
        last_vals - Ordered list of floats
        """
        if self.get_weights() == None or self.get_bias() == None:
            raise ValueError("Both weights and biases must be numerical in order to update the value.")
        weights_vec = np.array(self.get_weights())
        vals_vec = np.array(last_vals)
        bias = float(self.get_bias())
        self.pre_value = np.dot(weights_vec, vals_vec) + bias

    def update_value(self, last_vals, activation):
        """
        activation - Function taking in floats and outputting floats
        """
        self.update_pre_value(last_vals=last_vals)
        if self.get_pre_value() == None:
            return ValueError("Pre-value is not numerical so cannot compute or set a value.")
        pre_val = self.get_pre_value()
        self.value = activation(pre_val)

    def sigmoid(x):
        return 1/(1 + np.exp(-x))

class Input_Node(Node):

    def __init__(self, value):
        self.value = value

class Input_Layer(object):

    def __init__(self, nodes):
        """
        nodes is a list of Input_Node types 
        """
        self.nodes = nodes




class Inside_Layer(object):

    def __init__(self, nodes, prev=None):
        """
        nodes - list of Node object that form the layer
        prev - Previous Layer object in the network
        next - Next Layer object in the network
        """
        length_list = [len(Node.get_weights(node)) for node in nodes]
        flag = False
        
        for l in length_list:
            if l != length_list[0]:
                flag = True
                break
        if flag:
            raise ValueError("Nodes in the layer do not all have the same number of weights.")
        
        if prev != None and length_list[0] != prev.length:
            raise ValueError("Number of weights is not consistent with the number of nodes in previous layer.")

        
        # Define length attribute
        self.length = len(nodes)

        # Define nodes attribute
        self.nodes = nodes

        # Define previous layer attribute 
        self.prev = prev

    def get_length(self):
        return self.length

    def get_weight_matrix(self):
        weight_list = [Node.get_weights(node) for node in self.get_nodes()]
        return np.array(weight_list)

    def get_prev(self):
        return self.prev

    def get_nodes(self):
        return self.nodes

    def get_pre_values(self):
        nodes = self.get_nodes()
        return [Node.get_pre_value(node) for node in nodes]

    def get_values(self):
        nodes = self.get_nodes()
        return [Node.get_value(node) for node in nodes]

    def set_prev(self, layer):
        self.prev = layer

    def update_pre_values(self):
        if self.prev == None:
            raise ValueError("Previous Layer does not exist so cannot update values.")
        weight_matrix = self.get_weight_matrix()
        last_vals = [Node.get_value(node) for node in self.get_prev().get_nodes()]
        last_vec = np.reshape(np.array(last_vals), (-1,1))
        bias_list = [Node.get_bias(node) for node in self.get_prev().get_nodes()]
        bias_vec = np.reshape(bias_list, (-1,1))
        pre_values = np.matmul(weight_matrix, last_vec) + bias_vec
        for i, node in enumerate(self.get_nodes()):
            Node.set_pre_value(node, pre_value=pre_values[i])

    def update_values(self, activation):
        self.update_pre_values()
        for node in self.get_nodes():
            Node.update_value(node, activation)


node1 = Node(weights=[1,2,1], bias=-1, pre_value=0.5, value=0.3)
node2 = Node(weights=[-1,0.5,1], bias=1, pre_value=0.5, value=0.2)
node3 = Node(weights=[0,9,5], bias=0, pre_value=0.2, value=0.1)
node4 = Node(weights=[-1,3,1], bias=-1, pre_value=0.1, value=0.1)
node5 = Node(weights=[4,0.5,0.9], bias=0, pre_value=0.1, value=0.2)
node6 = Node(weights=[1,2,5], bias=0, pre_value=0.3, value=0.4)

layer1 = Inside_Layer(nodes=[node1,node2,node3])
layer2 = Inside_Layer(nodes=[node4,node5,node6])



layer2.set_prev(layer1)


# node1.update_value(activation=Node.sigmoid)
# node2.update_value(activation=Node.sigmoid)
# node3.update_value(activation=Node.sigmoid)
# node4.update_value(activation=Node.sigmoid)
# node5.update_value(activation=Node.sigmoid)
# node6.update_value(activation=Node.sigmoid)

class Network(object):

    def __init__(self, layers):
        """
        layers - List of Layer type object
        """
        # if type(layers[0]) != Input_Node:
        #     raise ValueError("First Layer must be of type Input")
        
        self.layers = layers

    def get_layers(self):
        return self.layers

    def get_layer_gradients(self, index, xgradients, activation_deriv):

        layer = self.get_layers()[index]
        next_layer = self.get_layers()[index+1]

        # Get values and pre-values of the current layer
        pre_values = Inside_Layer.get_pre_values(layer)
        values = Inside_Layer.get_values(layer)


        weight_matrix = Inside_Layer.get_weight_matrix(next_layer)

        # Compute the new, updated hgradients
        hgrads = np.array(xgradients) @ weight_matrix

        # Compute the new, updated xgradients
        xgrads = activation_deriv(np.array(pre_values)) * hgrads

        # Compute the beta gradients for the new layer
        betagrads = np.reshape(np.array(xgradients), (-1,1)) @ np.reshape(np.array(values), (1,-1))

        # Compute the mu gradient for the new layer
        mugrads = xgradients

        return xgrads, betagrads, mugrads


network = Network([layer1,layer2])
xgradients = [1,1,1]
activation_deriv = lambda x: x+1
index = 0
a, b, c = network.get_layer_gradients(index=index, xgradients=xgradients, activation_deriv=activation_deriv)
print(a)
print(b)
print(c)








    


        