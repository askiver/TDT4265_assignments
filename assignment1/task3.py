import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    predictions = model.forward(X)
    predictions = np.argmax(predictions, axis=1)
    targets = np.argmax(targets, axis=1)
    accuracy = np.mean(predictions == targets)
    return accuracy


class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        output = self.model.forward(X_batch)
        self.model.backward(X_batch, output, Y_batch)
        self.model.w = self.model.w - self.learning_rate * self.model.grad
        loss = cross_entropy_loss(Y_batch, output)
        return loss

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
            accuracy_train (float): Accuracy on train dataset
            accuracy_val (float): Accuracy on the validation dataset
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(self.Y_val, logits)

        accuracy_train = calculate_accuracy(
            self.X_train, self.Y_train, self.model)
        accuracy_val = calculate_accuracy(
            self.X_val, self.Y_val, self.model)
        return loss, accuracy_train, accuracy_val


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.

    # Intialize model
    model = SoftmaxModel(l2_reg_lambda)
    # Train model
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    plt.ylim([0.2, .8])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task3b_softmax_train_loss.png")
    plt.show()

    # Plot accuracy
    plt.ylim([0.89, .93])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3b_softmax_train_accuracy.png")
    plt.show()

    # Train a model with L2 regularization (task 4b)

    model1 = SoftmaxModel(l2_reg_lambda=1.0)
    trainer = SoftmaxTrainer(
        model1, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg01, val_history_reg01 = trainer.train(num_epochs)
    # You can finish the rest of task 4 below this point.

    # Plotting of softmax weights (Task 4b)
    # function that prepares the weights for visualization
    def visualize_softmax_weights(model: SoftmaxModel):
        # Let's exclude the bias term and reshape the weights
        weights = model.w[:-1, :].T  # Assuming the last weight is the bias term
        weights_reshaped = weights.reshape(10, 28, 28)

        # Normalize the weights for better visualization
        # Here we use a different approach for normalization to spread out the intensities
        weights_min = weights_reshaped.min()
        weights_max = weights_reshaped.max()
        weights_normalized = (weights_reshaped - weights_min) / (weights_max - weights_min)

        # Stack the images horizontally
        weights_horizontal = np.hstack(weights_normalized)
        return weights_horizontal
    weights = []
    for models in [model, model1]:
        weights.append(visualize_softmax_weights(models))

    plt.imsave("task4b_softmax_weight.png", np.vstack(weights), cmap="gray")

    # Plotting of accuracy for difference values of lambdas (task 4c)
    l2_lambdas = [1, .1, .01, .001]
    models = [SoftmaxModel(l2_reg_lambda=l) for l in l2_lambdas]
    trainers = [SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    ) for model in models]
    histories = [(trainer.train(num_epochs), trainer.model.l2_reg_lambda) for trainer in trainers]
    plt.ylim([0.80, .93])
    for history in histories:
        # Plot accuracy
        utils.plot_loss(history[0][1]["accuracy"], "Validation Accuracy Lambda: " + str(history[1]))
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task4c_l2_reg_accuracy.png")
    plt.close()

    # Task 4e - Plotting of the l2 norm for each weight
    l2_norms = [np.sqrt(np.sum(np.square(model.w))) for model in models]
    lambdas = [model.l2_reg_lambda for model in models]
    plt.plot(lambdas, l2_norms, "o")
    plt.xlabel("Lambda")
    plt.ylabel("L2 Norm of weights")
    plt.legend()
    plt.savefig("task4e_l2_reg_norms.png")


if __name__ == "__main__":
    main()
