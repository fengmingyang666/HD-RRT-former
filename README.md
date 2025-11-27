# HD-RRT-former

## Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy
- SciPy
- PyBullet
- tqdm
- wandb (optional, for training visualization)

## Quick Start

### 1. Demonstration

```bash
python demo.py
```

The demonstration script provides a simple interface to run the pre-trained model. By default, it uses the `ckpt/model_6D.pt` model for 6D space path planning.

### 2. Dataset Generation

Dataset can be downloaded from the following link:
[Download Dataset](https://drive.google.com/file/d/1yDZS8oNZJiot6dbN7IJVbqbZBpjgNW9E/view?usp=drive_link)

Unzip the downloaded dataset file and place it in the `data/` directory.

```bash
python dataset_generation.py
```

The dataset generation process simulates the robotic arm in a PyBullet environment and generates path planning data using the RRT-Connect algorithm. Generated datasets will be saved to the `data/` directory, with the default output being `dataset_6D.npy`.

The dataset includes:
- Path planning tree nodes
- Next sampling points
- Environment maps

### 3. Model Training

You can find the pre-trained model checkpoint in the `ckpt/` directory.

```bash
python train.py
```

The training process uses the previously generated dataset to train the Transformer model. Key parameter settings:
- Dataset path: `data/dataset_6D.npy`
- Batch size: 128
- Training epochs: 2000
- Learning rate: 1e-4
- Validation split: 10%

During training, checkpoints will be automatically saved to the `ckpt/` directory every 10 epochs, and the best validation model will be recorded.

### 4. Model Testing

```bash
python test.py --method rrt_connect --seed 2025 --episode 500 --alpha 1.0 --max_iters 2000 --step_size 0.2 --model "ckpt/model_7D.pt" --D '7D'
```

The testing script loads the trained model and evaluates path planning performance in new environments. The testing process will:
- Run path planning in environments with different obstacle configurations

