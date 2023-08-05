import torch
from torchrl.modules import MLP
import numpy as np


def prepare_training_data(env, num_samples):
    inputs = []
    targets = []

    for _ in range(num_samples):
        observation, _, _, _ = env.reset()
        inputs.append(observation)

        done = False
        while not done:
            action = env.action_space.sample()
            observation, reward, done, _ = env.step(action)
            targets.append([reward])

    return np.array(inputs), np.array(targets)


def prepare_validation_data(env, num_samples):
    inputs = []
    targets = []

    for _ in range(num_samples):
        observation, _, _, _ = env.reset()
        inputs.append(observation)

        done = False
        while not done:
            action = env.action_space.sample()
            observation, reward, done, _ = env.step(action)
            targets.append([reward])

    return np.array(inputs[:num_samples]), np.array(targets[:num_samples])


def compute_loss(outputs, targets):
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(outputs, targets)
    return loss


def train(env):
    # Set up hyperparameters
    input_size = 3  # Observation space size of the environment
    output_size = 1  # Single output value representing the reward
    depth = 2
    num_cells = 32
    learning_rate = 0.001
    num_epochs = 1000
    batch_size = 32
    validation_samples = 100

    # Create an instance of the MLP class
    mlp = MLP(in_features=input_size, out_features=output_size, depth=depth, num_cells=num_cells)

    # Define the optimizer
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)

    # Prepare the training data
    train_inputs, train_targets = prepare_training_data(env, num_samples=100)

    # Prepare the validation data
    val_inputs, val_targets = prepare_validation_data(env, num_samples=validation_samples)

    # Training loop
    for epoch in range(num_epochs):
        # Shuffle the training data
        indices = np.random.permutation(len(train_inputs))
        train_inputs = train_inputs[indices]
        train_targets = train_targets[indices]

        # Mini-batch training
        for i in range(0, len(train_inputs), batch_size):
            batch_inputs = train_inputs[i:i + batch_size]
            batch_targets = train_targets[i:i + batch_size]

            # Convert to tensors
            batch_inputs_tensor = torch.tensor(batch_inputs, dtype=torch.float32)
            batch_targets_tensor = torch.tensor(batch_targets, dtype=torch.float32)

            # Forward pass
            outputs = mlp(batch_inputs_tensor)

            # Compute the loss
            loss = compute_loss(outputs, batch_targets_tensor)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        val_inputs_tensor = torch.tensor(val_inputs, dtype=torch.float32)
        val_targets_tensor = torch.tensor(val_targets, dtype=torch.float32)
        val_outputs = mlp(val_inputs_tensor)
        val_loss = compute_loss(val_outputs, val_targets_tensor)

        # Print the epoch and validation loss
        print(f"Epoch: {epoch+1}/{num_epochs}, Validation Loss: {val_loss.item():.4f}")
