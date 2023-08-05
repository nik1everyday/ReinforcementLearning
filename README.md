
# Energy Environment RL Training

This project demonstrates the training of a reinforcement learning (RL) agent using an energy environment. The RL agent uses a multi-layer perceptron (MLP) model to learn to maximize rewards in the energy environment.

## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/nik1everyday/ReinforcementLearning.git
   cd ReinforcementLearning
   ```

2. Create a virtual environment (optional but recommended):

   ```shell
   python -m venv env
   source env/bin/activate  # Linux/Mac
   env\Scripts\activate  # Windows
   ```

3. Install the dependencies:

   ```shell
   poetry install
   ```

## Usage

1. Create and activate the virtual environment (if not done already):

   ```shell
   source env/bin/activate  # Linux/Mac
   env\Scripts\activate  # Windows
   ```

2. Run the main script:

   ```shell
   python main.py
   ```

   This will start the training process. The script will train the RL agent using the energy environment and output the validation loss for each epoch.

3. Monitor the training progress:

   The training progress will be displayed in the console. You can monitor the validation loss and observe how it changes over the epochs.

4. Customize the training parameters (optional):

   You can modify the hyperparameters and other settings in the `train.py` file to customize the training process. Adjust the values according to your requirements.

## Project Structure

- `main.py`: The main script to run the RL training process.
- `train.py`: Contains the training logic, including data preparation, model creation, and training loop.
- `src/environment.py`: Defines the `EnergyEnvironment` class that represents the energy environment.
- `README.md`: This file.

