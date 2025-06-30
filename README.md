# Hands_On_NeRF
# NeRF: Neural Radiance Fields Implementation

## Overview
This project is a hands-on implementation of the Neural Radiance Fields (NeRF) paper: "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis". It learns a continuous 5D representation of a scene to synthesize novel views with high fidelity.

This implementation includes:
- A fully-connected (MLP) network for representing the scene.
- Positional encoding to capture high-frequency details.
- A hierarchical sampling strategy with a "coarse" and a "fine" network.
- Volume rendering to synthesize images from the neural representation.

## File Structure
- `train.py`: The main script to start the training process.
- `utils.py`: Contains the core components of the NeRF model, data loading classes, and rendering utility functions.

## Implementation Details

### `utils.py`
This file contains the building blocks of the NeRF model.

- **`NeRF` class**: The main network, an MLP that takes a 5D coordinate (3D location + 2D viewing direction) and outputs a volume density (sigma) and an RGB color.
- **`Embedder` class**: Implements positional encoding for the input coordinates, which helps the model learn high-frequency functions.
- **`ViewDependentHead` class**: The final part of the NeRF network that takes the processed features and the view direction to output the final RGB color.
- **`DatasetProvider` & `NeRFDataset`**: Classes responsible for loading the synthetic dataset and preparing the rays and ground-truth pixels for training.
- **Rendering Functions**:
    - `sample_rays`: Generates rays from camera poses.
    - `predict_to_rgb`: Performs volume rendering along the rays to compute the final pixel colors.
    - `sample_pdf`: Implements the hierarchical sampling by sampling more points from regions with higher expected contribution.

### `train.py`
This script orchestrates the training process.

- It initializes two `NeRF` models: a `coarse` model and a `fine` model, as described in the paper.
- It uses the Adam optimizer and Mean Squared Error (MSE) loss between the rendered and ground-truth pixels.
- The `render_rays` function implements the core rendering logic, which first samples points along each ray, queries the `coarse` network, and then uses the output to perform a more informed sampling for the `fine` network.
- The final loss is the sum of the losses from both the coarse and fine predictions.

## How to Run

1.  **Install dependencies:**
    ```bash
    pip install torch numpy imageio opencv-python
    ```
    The dataset is nerf_synthetic, can be downloaded in https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4
2.  **Download the data:**
    The code is currently configured to use the `lego` dataset from the `nerf_synthetic` dataset. Make sure this dataset is available in the `NeRF/nerf_synthetic/lego` directory.
3.  **Start training:**
    ```bash
    python train.py
    ```
    The script will start training the model and print the loss for each epoch. You can modify the `root` and `transforms_file` variables in `train.py` to use a different dataset.
