import torch


def train_model_generic(model, train_loader, test_loader, optimizer, loss_func, device, epochs):
    """
    generic training procedure
    :param model: model to train
    :param train_loader: training data loader
    :param test_loader: test data loader
    :param optimizer: optimizer includes learning rate and other specifications
    :param loss_func: loss function
    :param device: cpu, mps, gpu device to train on
    :param epochs: number of epochs to train
    :return: train and test losses
    """
    train_losses = []
    test_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # Handle models that return a tuple (e.g., (output, latent)) by using the first element.
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            loss = loss_func(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch[0].to(device)
                outputs = model(inputs)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                loss = loss_func(outputs, inputs)
                running_loss += loss.item() * inputs.size(0)
        epoch_test_loss = running_loss / len(test_loader.dataset)
        test_losses.append(epoch_test_loss)

        print(f"Epoch [{epoch}/{epochs}] Train Loss: {epoch_train_loss:.4f} | Test Loss: {epoch_test_loss:.4f}")

    return train_losses, test_losses
