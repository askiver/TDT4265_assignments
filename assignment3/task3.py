import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from dataloaders import load_cifar10
from custom_trainer import Custom_trainer
from trainer import compute_loss_and_accuracy


class Incredible_model(nn.Module):
    def __init__(self, image_channels, num_classes):
        """
        Is called when model is initialized.
        Args:
            image_channels. Number of color channels in image (3)
            num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        # TODO: Implement this function (Task  2a)
        num_filters = 128 # Set number of filters in first conv layer
        self.num_classes = num_classes
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="zeros",
                bias=False,
            ),
            nn.BatchNorm2d(num_filters),
            # nn.ReLU(),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters*2,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="zeros",
                bias=False,
            ),
            nn.BatchNorm2d(num_filters*2),
            # nn.ReLU(),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=num_filters*2,
                out_channels=num_filters*4,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="zeros",
                bias=False,
            ),
            nn.BatchNorm2d(num_filters*4),
            # nn.ReLU(),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=num_filters * 4,
                out_channels=num_filters * 8,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="zeros",
                bias=False,
            ),
            nn.BatchNorm2d(num_filters * 8),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * 2 * 8 * num_filters, 256, bias=False),
            nn.BatchNorm1d(256),
            # nn.ReLU(),
            nn.GELU(),

            nn.Linear(256, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        x = self.feature_extractor(x)
        x = self.classifier(x)
        out = x
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (
            batch_size,
            self.num_classes,
        ), f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return x


def create_plots(trainer: Custom_trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(
        trainer.train_history["loss"], label="Training loss", npoints_to_average=10
    )
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


def compare_trainers(trainer1, trainer2):

    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Loss Before modification")
    utils.plot_loss(
        trainer1.train_history["loss"], label="Training loss", npoints_to_average=10
    )
    utils.plot_loss(trainer1.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Loss after modification")
    utils.plot_loss(
        trainer2.train_history["loss"], label="Training loss", npoints_to_average=10
    )
    utils.plot_loss(trainer2.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{'Task3d'}_plot.png"))
    plt.show()




def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result!
    utils.set_seed(0)
    print(f"Using device: {utils.get_device()}")
    epochs = 10
    batch_size = 64
    learning_rate = 0.0001
    early_stop_count = 10
    dataloaders = load_cifar10(batch_size, augment_data=True)
    worse_dataloaders = load_cifar10(batch_size, augment_data=False)
    model = Incredible_model(image_channels=3, num_classes=10)
    model2 = Incredible_model(image_channels=3, num_classes=10)
    trainer = Custom_trainer(
        batch_size, learning_rate, early_stop_count, epochs, model, dataloaders
    )
    trainer.train()
    #create_plots(trainer, "task3")
    trainer2 = Custom_trainer(
        batch_size, learning_rate, early_stop_count, epochs, model2, worse_dataloaders
    )
    trainer2.train()

    compare_trainers(trainer2, trainer)

    # Print loss and accuracy for all three datasets
    """
    loss, accuracy = compute_loss_and_accuracy(trainer.dataloader_train, trainer.model, trainer.loss_criterion)
    print("Train loss:", loss)
    print("Train accuracy:", accuracy)
    loss, accuracy = compute_loss_and_accuracy(trainer.dataloader_val, trainer.model, trainer.loss_criterion)
    print("Validation loss:", loss)
    print("Validation accuracy:", accuracy)
    loss, accuracy = compute_loss_and_accuracy(trainer.dataloader_test, trainer.model, trainer.loss_criterion)
    print("Test loss:", loss)
    print("Test accuracy:", accuracy)
    """


if __name__ == "__main__":
    main()
