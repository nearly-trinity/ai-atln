import nn


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.
        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.
        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(x, self.get_weights())

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.
        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        dotProd = nn.as_scalar(self.run(x))
        return -1 if dotProd < 0 else 1

    def train(self, dataset):
        loop = True
        while loop:
            loop = False
            for x, y in dataset.iterate_once(1):
                if self.get_prediction(x) != nn.as_scalar(y):
                    nn.Parameter.update(self.w, x, nn.as_scalar(y))
                    loop = True


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 25
        self.num_neurons_hidden_layer = 50

        # layer 1
        self.w_1 = nn.Parameter(
            1, self.num_neurons_hidden_layer)  # weight vector 1
        self.b_1 = nn.Parameter(
            1, self.num_neurons_hidden_layer)  # bias vector 1

        # output layer
        self.output_w = nn.Parameter(self.num_neurons_hidden_layer, 1)
        self.output_b = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # Calculate layer 1 weights
        trans_1 = nn.Linear(x, self.w_1)

        # Calculate layer 1 weights with biases
        trans_1_bias = nn.AddBias(trans_1, self.b_1)

        # Convert to ReLU
        relu_1 = nn.ReLU(trans_1_bias)

        # Use ReLU to compute output weight
        trans_2 = nn.Linear(relu_1, self.output_w)

        # Get output weights with biases
        trans_2_bias = nn.AddBias(trans_2, self.output_b)

        # Return output unit
        return trans_2_bias

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.
``
        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        # creates the predictions for y (yhat)
        yhat = self.run(x)

        # calculates the error rate (mean squared error) of the predictions vs the true values
        return nn.SquareLoss(yhat, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # initial learning rate
        adjusted_rate = -0.2

        # loop for performing gradient updates
        while 1 == 1:
            for row_vect, label in dataset.iterate_once(self.batch_size):

                # calculate loss using the information from the dataset
                loss = self.get_loss(row_vect, label)

                # creates a list of the layer parameters
                params = [self.w_1, self.b_1, self.output_w, self.output_b]

                # calculates gradients using the loss and the parameters
                gradients = nn.gradients(loss, params)

                learning_rate = min(-0.01, adjusted_rate)

                # Compute updates with the new learning rate
                self.w_1.update(gradients[0], learning_rate)
                self.output_w.update(gradients[2], learning_rate)
                self.b_1.update(gradients[1], learning_rate)
                self.output_b.update(gradients[3], learning_rate)

            # update learning rate
            adjusted_rate += .02

            # recalculate true loss
            loss = self.get_loss(nn.Constant(dataset.x),
                                 nn.Constant(dataset.y))

            # break loop when error rate is low enough
            if nn.as_scalar(loss) < 0.008:
                return


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here

        # set hyperparameters
        self.batch_size = 100
        self.hidden_layer_size = 100
        self.learning_rate = - 0.5
        self.number_hidden_layers = 2

        # initialize weights and bias for first layer of input
        self.w_1 = nn.Parameter(784, self.hidden_layer_size)
        # 1 bias value goes into each of the nodes in the hidden layer
        self.b_1 = nn.Parameter(1, self.hidden_layer_size)

        # initialize weights and bias for 2nd layer of input
        # size of input x size of output (both size of hidden layer)
        self.w_2 = nn.Parameter(self.hidden_layer_size, self.hidden_layer_size)
        self.b_2 = nn.Parameter(1, self.hidden_layer_size)

        # initialize weights and bias for 3rd layer (output)
        # size of input (hidden layer) x size of output (10)
        self.w_3 = nn.Parameter(self.hidden_layer_size, 10)
        self.b_3 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

        # calculations layer 1
        # matrix multiplication of features x weights
        xw_1 = nn.Linear(x, self.w_1)
        xw_1_b = nn.AddBias(xw_1, self.b_1)  # add in bias
        ReLU_1 = nn.ReLU(xw_1_b)  # perform activation function (ReLU)

        # calculations layer 2, output from layer 1 as input
        xw_2 = nn.Linear(ReLU_1, self.w_2)
        xw_2_b = nn.AddBias(xw_2, self.b_2)
        ReLU_2 = nn.ReLU(xw_2_b)

        # calculation layer 3 (last layer, no ReLU)
        xw_3 = nn.Linear(ReLU_2, self.w_3)
        xw_3_b = nn.AddBias(xw_3, self.b_3)

        return xw_3_b

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(x)

        # calculates a batched softmax loss
        loss = nn.SoftmaxLoss(predicted_y, y)

        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        training = True

        while training:

            for input, output in dataset.iterate_once(self.batch_size):

                loss = self.get_loss(input, output)
                # takes in loss and parameters, returns loss for each parameter
                grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3 = nn.gradients(
                    loss, [self.w_1, self.b_1, self.w_2, self.b_2, self.w_3, self.b_3])

                # updates parameters using direction (gradients) and multiplier (learning rate)
                self.w_1.update(grad_w1, self.learning_rate)
                self.b_1.update(grad_b1, self.learning_rate)
                self.w_2.update(grad_w2, self.learning_rate)
                self.b_2.update(grad_b2, self.learning_rate)
                self.w_3.update(grad_w3, self.learning_rate)
                self.b_3.update(grad_b3, self.learning_rate)

            accuracy = dataset.get_validation_accuracy()
            if accuracy >= .975:
                return

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        self.w_1 = nn.Parameter(self.num_chars, 100)
        self.b_1 = nn.Parameter(1, 100)
        self.w_2 = nn.Parameter(100, 100)
        self.b_2 = nn.Parameter(1, 100)
        self.w_1_hidden = nn.Parameter(100, 100)
        self.b_1_hidden = nn.Parameter(100, 1)
        self.w_2_hidden = nn.Parameter(100, 100)
        self.b_2_hidden = nn.Parameter(1, 100)
        self.w_end = nn.Parameter(100, 5)
        self.b_end = nn.Parameter(1, 5)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        for i in range(len(xs)):
            if i == 0:
                zIfZero = nn.AddBias(nn.Linear(xs[i],self.w_1), self.b_1)
                activation = nn.ReLU(zIfZero)
                hidden = nn.AddBias(nn.Linear(activation,self.w_2), self.b_2)
            else:
                z = nn.ReLU(nn.Add(nn.Linear(xs[i],self.w_1), nn.Linear(hidden, self.w_1_hidden)))
                hidden = nn.ReLU(nn.AddBias(nn.Linear(z ,self.w_2_hidden),self.b_2_hidden))
 
        return nn.AddBias(nn.Linear(hidden,self.w_end),self.b_end)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs),y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batchSize = 100
        iteration = 0
        accuracy = float(0)
        while accuracy < .85 and iteration < 40:
            iteration += 1
            # update every layer of biases and weights, hidden or otherwise
            for x,y in dataset.iterate_once(batchSize):
                gradientW1, gradientB1, gradientW2, gradientB2, gradientW1_hidden, gradientB1_hidden, gradientW2_hidden, gradientB2_hidden, gradientWEnd, gradientBEnd \
                = nn.gradients(self.get_loss(x,y), [self.w_1, self.b_1, self.w_2, self.b_2, self.w_1_hidden, self.b_1_hidden, self.w_2_hidden, self.b_2_hidden, self.w_end, self.b_end])
                self.w_1.update(gradientW1, -0.15)
                self.b_1.update(gradientB1, -0.15)
                self.w_2.update(gradientW2, -0.15)
                self.b_2.update(gradientB2, -0.15)
                self.w_1_hidden.update(gradientW1_hidden, -0.15)
                self.b_1_hidden.update(gradientB1_hidden, -0.15)
                self.w_2_hidden.update(gradientW2_hidden, -0.15)
                self.b_2_hidden.update(gradientB2_hidden, -0.15)
                self.w_end.update(gradientWEnd, -0.15)
                self.b_end.update(gradientBEnd, -0.15)
                accuracy = dataset.get_validation_accuracy()
            

