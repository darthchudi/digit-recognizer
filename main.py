import torch.nn as nn
import torch
import torchsummary
from torch.utils.data import DataLoader, random_split
from network import Network
from trainer import Trainer
from dataset import DigitDataset

def build_network():
    # Split the dataset into training and eval datasets so we can run evals on unseen data
    base_dataset = DigitDataset("./data/train.csv", normalise_pixels=True)
    training_dataset, eval_dataset = random_split(base_dataset, [0.95, 0.05])

    # Setup the dataloaders
    train_dataloader = DataLoader(training_dataset, shuffle=True, batch_size=600, num_workers=2)
    eval_dataloader = DataLoader(eval_dataset, shuffle=True, batch_size=150, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # initialise the model and train it
    network = Network(input_dim=28 * 28, output_dim=10, use_conv2d=True)
    network.to(device)

    print(torchsummary.summary(network, input_size=(1, 28, 28)))

    trainer = Trainer(network,
                      num_epochs=30,
                      train_batch_limit=-1,
                      eval_batch_limit=-1,
                      learning_rate=0.1,
                      eval_accuracy=0.98,
                      device=device,
                      train_dataloader=train_dataloader,
                      eval_dataloader=eval_dataloader,
                      )
    trainer.train()

    # Save the model's weights after training
    torch.save(network.state_dict(), "digit-recogniser.pth")
    print("✨ Successfully trained model and saved weights")

    return network


if __name__ == "__main__":
    _ = build_network()


