import torch.nn as nn
import torch

# Define a class for orchestrating the network's training
class Trainer():
    def __init__(self, network, num_epochs, learning_rate, train_batch_limit=10, eval_batch_limit=2, eval_accuracy=None,
                 device="cpu", train_dataloader=None, eval_dataloader=None):
        self.network = network
        self.num_epochs = num_epochs
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimiser = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)
        self.train_batch_limit = train_batch_limit
        self.eval_batch_limit = eval_batch_limit
        self.eval_accuracy = eval_accuracy
        self.device = device
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

    def run_epoch_training(self, epoch):
        print(f"==== \n[epoch #{epoch + 1} ]\n====")

        total_loss = 0
        num_accurate_predictions = 0
        total_samples = 0
        total_batches = 0

        self.network.train()
        for batch, (x, y) in enumerate(self.train_dataloader):
            total_batches += 1
            total_samples += len(x)

            x = x.to(self.device)
            y = y.to(self.device)

            # Clear previous gradients
            self.optimiser.zero_grad()

            # Run inference
            logits = self.network(x)
            logits = logits.type(torch.float32)

            # Compute loss, gradients and update the weights
            loss = self.loss_fn(logits, y)
            loss.backward()
            self.optimiser.step()

            total_loss += loss.item()

            _, prediction = logits.max(dim=1)
            num_accurate_predictions_in_batch = (prediction == y).type(torch.float32).sum().item()
            num_accurate_predictions += num_accurate_predictions_in_batch

            if batch == self.train_batch_limit:
                break

        accuracy = num_accurate_predictions / total_samples
        total_loss /= total_batches
        print(
            f"[Test] Accuracy %: {accuracy * 100}%. Loss: {total_loss}. Predicted {num_accurate_predictions} / {total_samples} items.")

        return accuracy, total_loss

    def run_epoch_eval(self, epoch):
        print("==Evals==")

        total_loss = 0
        num_accurate_predictions = 0
        total_samples = 0
        total_batches = 0

        self.network.eval()
        with torch.no_grad():
            for batch, (x, y) in enumerate(self.eval_dataloader):
                total_batches += 1
                total_samples += len(x)

                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.network(x)
                logits = logits.type(torch.float32)

                loss = self.loss_fn(logits, y)
                total_loss += loss.item()

                _, prediction = logits.max(dim=1)
                num_accurate_predictions_in_batch = (prediction == y).type(torch.float32).sum().item()
                num_accurate_predictions += num_accurate_predictions_in_batch

                if batch == self.eval_batch_limit:
                    break

            accuracy = num_accurate_predictions / total_samples
            total_loss /= total_batches
            print(
                f"[Eval] Accuracy %: {accuracy * 100}%. Loss: {total_loss}. Predicted {num_accurate_predictions} / {total_samples} items.")

            return accuracy, total_loss

    def train(self):
        if self.train_dataloader is None or self.eval_dataloader is None:
            raise Exception("missing dataloader for training or eval dataset")

        for epoch in range(self.num_epochs):
            self.run_epoch_training(epoch)
            eval_accuracy, _ = self.run_epoch_eval(epoch)

            if self.eval_accuracy is not None and eval_accuracy >= self.eval_accuracy:
                print(f"🎉 Reached {self.eval_accuracy * 100}% eval accuracy. Ending training")
                break

