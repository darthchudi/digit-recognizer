import torch.nn as nn
import pandas as pd
import os.path
import torch
import torchsummary
from torch.utils.data import DataLoader, random_split
from network import Network
from trainer import Trainer
from dataset import DigitDataset

def build_network(device):
    # Split the dataset into training and eval datasets so we can run evals on unseen data
    base_dataset = DigitDataset("./data/train.csv", normalise_pixels=True)
    training_dataset, eval_dataset = random_split(base_dataset, [0.95, 0.05])

    # Setup the dataloaders
    train_dataloader = DataLoader(training_dataset, shuffle=True, batch_size=600, num_workers=2)
    eval_dataloader = DataLoader(eval_dataset, shuffle=True, batch_size=150, num_workers=2)

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
    torch.save(network.state_dict(), "digit-recognizer.pth")
    print("✨ Successfully trained model and saved weights")

    return network


def run_submission_inference(device):
    # Confirm that the model's weights are in the directory
    if os.path.isfile("digit-recognizer.pth") is False:
        raise Exception("model checkpoint file not found")

    # Load the weights from the model
    network = Network(input_dim=28 * 28, output_dim=10, use_conv2d=True)
    network.to(device)

    network.load_state_dict(torch.load("digit-recognizer.pth"))
    network.eval()

    # Setup the test dataset and dataloader for submission
    test_dataset = DigitDataset("./data/test.csv", normalise_pixels=True)
    test_dataloader = DataLoader(test_dataset, batch_size=600, num_workers=2)

    # Compute classifications for the test dataset
    results = []
    with torch.no_grad():
        for batch, (x, y) in enumerate(test_dataloader):
            x = x.to(device)

            logits = network(x)
            logits = logits.type(torch.float32)

            _, prediction = logits.max(dim=1)
            results = results + prediction.tolist()

    # Save the computed results in a csv file
    raw_submission_df = []
    for index, label in enumerate(results):
        # Create a row in the format (ImageId,Label) for each result
        row = [index + 1, label]
        raw_submission_df.append(row)

    submission_df = pd.DataFrame(raw_submission_df, columns=["ImageId", "Label"])
    submission_df.to_csv("./data/submission.csv", index=False)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _ = build_network(device)
    run_submission_inference(device)



