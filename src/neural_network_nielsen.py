import numpy as np

class Network(object):

    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.weights = [np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(x,1) for x in sizes[1:]]

    def get_sizes(self):
        return self.sizes

    def get_num_layers(self):
        return self.num_layers

    def get_weights(self):
        return self.weights

    def get_biases(self):
        return self.biases

    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def feedforward(self, a):
        """
        Returns the feedforward output of the network given an input a 
        """
        h = a
        biases = self.get_biases()
        weights = self.get_weights()
        for b, w in zip(biases, weights):
            x = np.matmul(w, h) + b
            h = sigmoid(x)
        return h

    



    


    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        
        # Loop through each epoch and for each epoch calculate gradients and 
        # as such a different weight and bias changes for each mini batch within the training data
        for epoch in range(epochs):
            print(f"Starting Epoch {epoch}...")
            # Randomly generate the batches by randomly splitting training tuples into 
            # mini_batch_sized sized batches
            batches = random_batch_split(training_data=training_data, mini_batch_size=mini_batch_size)
            for mini_batch in batches:
                self.update_mini_batch(mini_batch=mini_batch, eta=eta) 
            if test_data != None:
                inputs = [example[0] for example in test_data]
                output_activations = [self.feedforward(input) for input in inputs]
                y = [example[1] for example in test_data]
                differences = [vectorized_result(y) - activation for y, activation in zip(y, output_activations)]
                losses = [np.dot(np.transpose(diff), diff) for diff in differences]
                print(f"Epoch {epoch}: Test Loss is {(1/len(test_data)) * sum(losses)}, Test Accuracy is {self.evaluate(test_data=test_data)}")



    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.back_prop(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        biases = self.get_biases()
        weights = self.get_weights()
        self.biases = [bias - (eta * nabla)/len(mini_batch) for bias, nabla in zip(biases, nabla_b)]
        self.weights = [weight - (eta*nabla)/len(mini_batch) for weight, nabla in zip(weights, nabla_w)]


    def back_prop(self, x, y):

        sizes = self.get_sizes()
        biases = self.get_biases()
        weights = self.get_weights()
        size = self.get_num_layers()


        nabla_b = [np.zeros(b.shape) for b in biases]
        nabla_w = [np.zeros(w.shape) for w in weights]
        
        a = x
        activations = [x]
        zs = []
        for b, w in zip(biases, weights):
            z = np.matmul(w, a) + b
            a = sigmoid(z)
            zs.append(z)
            activations.append(a)

        delta = cost_derivative(y, activations[-1]) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.matmul(delta, np.transpose(activations[-2]))

        for l in range(2, size):
            weight_matrix = weights[-l+1]
            sp = sigmoid_prime(zs[-l])
            delta = np.matmul(np.transpose(weight_matrix), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.matmul(delta, np.transpose(activations[-l-1]))
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        inputs = [example[0] for example in test_data]
        output_activations = [self.feedforward(input) for input in inputs]
        y = [example[1] for example in test_data]
        results = [np.argmax(output)==int(y) for output, y in zip(output_activations, y)]
        return sum(results)/len(results)

    def confusion_matrix(self, test_data):
        """Return a 10 by 10 matrix where the 0 axis represents the true label value and 
        the 1 axis represents the predicted label value. 
        """
        matrix = np.zeros(shape=(10,10))
        inputs = [example[0] for example in test_data]
        output_activations = [self.feedforward(input) for input in inputs]
        predictions = [np.argmax(activations) for activations in output_activations]
        y = [example[1] for example in test_data]
        outcomes = zip(y, predictions)
        for outcome in outcomes:
            true, prediction = outcome
            matrix[true, prediction] = int(matrix[true, prediction]) + int(1)

        return matrix

    def get_predictions(self, test_data):
        
        inputs = [example[0] for example in test_data]
        output_activations = [self.feedforward(input) for input in inputs]
        predictions = [np.argmax(activations) for activations in output_activations]
        
        return predictions
        

    def save(self, filename):
        np.savez(
            filename,
            weights=np.array(self.weights, dtype=object),
            biases=np.array(self.biases, dtype=object)
        )

    def load(self, filename):
        data = np.load(filename, allow_pickle=True)
        self.weights = list(data["weights"])
        self.biases = list(data["biases"])





def cost_derivative(y, activations):
    return activations - y

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))

def random_batch_split(training_data, mini_batch_size):
        np.random.shuffle(training_data)
        batch_number = int(np.ceil(len(training_data)/mini_batch_size))
        batches = [training_data[i::batch_number] for i in range(batch_number)]
        return batches

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
