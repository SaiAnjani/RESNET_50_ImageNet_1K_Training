# ResNet-50 Training on ImageNet-1K

This repository contains code to train a ResNet-50 model on the ImageNet-1K dataset using PyTorch. The code is organized into multiple files for better modularity and readability.

## File Structure

- `data_loader.py`: Handles data loading and transformations.
- `model.py`: Defines the ResNet-50 model architecture.
- `train.py`: Contains the training loop.
- `test.py`: Contains the testing loop.
- `main.py`: The main script to run the training and testing.

## Prerequisites

- Python 3.6 or higher
- PyTorch
- torchvision
- tqdm
- TensorBoard

## Setup

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/resnet50-training.git
    cd resnet50-training
    ```

2. **Install the required packages**:
    ```sh
    pip install torch torchvision tqdm tensorboard
    ```

3. **Prepare the ImageNet-1K dataset**:
    - Download the ImageNet-1K dataset and organize it into the following structure:
      ```
      /path/to/imagenet/
      ├── train/
      │   ├── class1/
      │   │   ├── img1.jpg
      │   │   ├── img2.jpg
      │   │   └── ...
      │   ├── class2/
      │   │   ├── img1.jpg
      │   │   ├── img2.jpg
      │   │   └── ...
      │   └── ...
      └── val/
          ├── class1/
          │   ├── img1.jpg
          │   ├── img2.jpg
          │   └── ...
          ├── class2/
          │   ├── img1.jpg
          │   ├── img2.jpg
          │   └── ...
          └── ...
      ```

4. **Update the paths in the code**:
    - Update the paths to the ImageNet dataset in `data_loader.py`:
      ```python
      train_dir = '/path/to/imagenet/train'
      val_dir = '/path/to/imagenet/val'
      ```

## Running the Code

1. **Run the training script**:
    ```sh
    python main.py
    ```

2. **Monitor training with TensorBoard**:
    ```sh
    tensorboard --logdir=logs
    ```

## Notes

- Ensure you have sufficient disk space and memory to handle the ImageNet-1K dataset.
- Adjust the batch size and number of workers in `data_loader.py` based on your hardware capabilities.

## License

This project is licensed under the MIT License.