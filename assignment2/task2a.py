import numpy as np
import utils
import typing

np.random.seed(1)


def pre_process_images(X: np.ndarray, mean: float = 33.632238520408166, std: float = 78.97469229932484):
    """
    Args:
        std: single float value representing the standard deviation of all pixel values
        mean: single float value representing the mean of all pixel values
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784, f"X.shape[1]: {X.shape[1]}, should be 784"
    # TODO implement this function (Task 2a)
    X = (X - mean) / std
    # Add bias
    X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    return X


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape, \
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"

    # TODO implement this function (Task 3a)
    # Compute individual cross-entropy loss for all classes and samples
    cross_entropy = -np.sum(targets * np.log(outputs), axis=1)

    # Compute the average cross-entropy loss across all samples
    average_loss = np.mean(cross_entropy)

    return average_loss


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class SoftmaxModel:

    def __init__(
            self,
            # Number of neurons per layer
            neurons_per_layer: typing.List[int],
            use_improved_sigmoid: bool,  # Task 3b hyperparameter
            use_improved_weight_init: bool,  # Task 3a hyperparameter
            use_relu: bool,  # Task 3c hyperparameter
    ):
        np.random.seed(
            1
        )  # Always reset random seed before weight init to get comparable results.
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid
        self.use_relu = use_relu
        self.use_improved_weight_init = use_improved_weight_init

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and an output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            if self.use_improved_weight_init:
                w = np.random.normal(0, 1 / np.sqrt(prev), w_shape)
            else:
                w = np.random.uniform(-1, 1, w_shape)
            self.ws.append(w)
            prev = size
        self.grads = [None for _ in range(len(self.ws))]
        self.previous_grads = [0 for _ in range(len(self.ws))]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # TODO implement this function (Task 2b)
        # HINT: For performing the backward pass, you can save intermediate activations in variables in the forward pass.
        activation = X
        activations = [X]  # List to store all the activations, layer by layer
        zs = []  # List to store all the z vectors, layer by layer

        for i, w in enumerate(self.ws):
            z = np.dot(activation, w)
            zs.append(z)

            if i < len(self.ws) - 1:  # Not the last layer
                if self.use_relu and i > 0:  # ReLU for hidden layers
                    activation = np.maximum(0, z)
                elif self.use_improved_sigmoid:
                    activation = 1.7159 * np.tanh(2 / 3 * z)

                else:
                    activation = 1 / (1 + np.exp(-z))  # Sigmoid
            else:
                activation = softmax(z)  # Softmax for the final layer

            activations.append(activation)

        self.activations = activations
        self.zs = zs

        return activations[-1]  # Output of the network

    def backward(self, X: np.ndarray, outputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)
        assert (
                targets.shape == outputs.shape
        ), f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        # Compute the gradient on the output layer
        delta = outputs - targets  # For softmax and cross-entropy loss
        self.grads[-1] = np.dot(self.activations[-2].T, delta) / X.shape[0]

        # Backpropagate through hidden layers
        for l in range(len(self.neurons_per_layer) - 2, -1, -1):
            if self.use_relu:
                d_activation = self.zs[l] > 0
            elif self.use_improved_sigmoid:
                d_activation = (1.7159 * 2 / 3) * (1 - (np.tanh(2 / 3 * self.zs[l])) ** 2)
            else:
                d_activation = self.activations[l+1] * (1 - self.activations[l+1])

            print("d_activation shape:", d_activation.shape)
            print("delta shape:", delta.shape)
            print("w shape:", self.ws[l + 1].T.shape)

            delta = np.dot(delta, self.ws[l + 1].T) * d_activation
            if l == 0:
                self.grads[l] = np.dot(X.T, delta) / X.shape[0]  # Gradient for the first hidden layer
            else:
                self.grads[l] = np.dot(self.activations[l].T, delta) / X.shape[0]

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    identity = np.eye(num_classes)
    Y = identity[Y.reshape(-1)]
    return Y


def gradient_approximation_test(model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
    Numerical approximation for gradients. Should not be edited.
    Details about this test is given in the appendix in the assignment.
    """

    assert isinstance(X, np.ndarray) and isinstance(
        Y, np.ndarray
    ), f"X and Y should be of type np.ndarray!, got {type(X), type(Y)}"

    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                             model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon ** 1, (
                    f"Calculated gradient is incorrect. "
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n"
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n"
                    f"If this test fails there could be errors in your cross entropy loss function, "
                    f"forward function or backward function"
                )


def main():
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert (
            Y[0, 3] == 1 and Y.sum() == 1
    ), f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert (
            X_train.shape[1] == 785
    ), f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_relu = True
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init, use_relu
    )

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)


if __name__ == "__main__":
    main()
