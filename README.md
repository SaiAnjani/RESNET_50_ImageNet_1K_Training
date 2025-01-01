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

## Training Logs

- value of epoch is 0 Test Error: Accuracy: 7.9%, Avg loss: 11.382029
- value of epoch is 1 Test Error: Accuracy: 15.4%, Avg loss: 4.622222
- value of epoch is 2 Test Error: Accuracy: 25.0%, Avg loss: 3.720075
- value of epoch is 3 Test Error: Accuracy: 25.3%, Avg loss: 3.706519
- value of epoch is 4 Test Error: Accuracy: 31.9%, Avg loss: 3.246583
- value of epoch is 5 Test Error: Accuracy: 28.4%, Avg loss: 3.539784
- value of epoch is 6 Test Error: Accuracy: 33.0%, Avg loss: 3.209344
- value of epoch is 7 Test Error: Accuracy: 28.4%, Avg loss: 3.575979
- value of epoch is 8 Test Error: Accuracy: 33.6%, Avg loss: 3.177567
- value of epoch is 9 Test Error: Accuracy: 28.0%, Avg loss: 3.611734
- value of epoch is 10 Test Error: Accuracy: 29.9%, Avg loss: 3.442933
- value of epoch is 11 Test Error: Accuracy: 34.9%, Avg loss: 3.119334
- value of epoch is 12 Test Error: Accuracy: 30.1%, Avg loss: 3.449284
- value of epoch is 13 Test Error: Accuracy: 35.4%, Avg loss: 3.045673
- value of epoch is 14 Test Error: Accuracy: 35.0%, Avg loss: 3.078165
- value of epoch is 15 Test Error: Accuracy: 33.8%, Avg loss: 3.179208
- value of epoch is 16 Test Error: Accuracy: 30.2%, Avg loss: 3.490924
- value of epoch is 17 Test Error: Accuracy: 34.4%, Avg loss: 3.152208
- value of epoch is 18 Test Error: Accuracy: 37.0%, Avg loss: 2.968687
- value of epoch is 19 Test Error: Accuracy: 33.5%, Avg loss: 3.203178
- value of epoch is 20 Test Error: Accuracy: 34.9%, Avg loss: 3.102601
- value of epoch is 21 Test Error: Accuracy: 33.3%, Avg loss: 3.199428
- value of epoch is 22 Test Error: Accuracy: 34.9%, Avg loss: 3.085857
- value of epoch is 23 Test Error: Accuracy: 29.0%, Avg loss: 3.541592
- value of epoch is 24 Test Error: Accuracy: 35.1%, Avg loss: 3.089956
- value of epoch is 25 Test Error: Accuracy: 29.8%, Avg loss: 3.481666
- value of epoch is 26 Test Error: Accuracy: 31.0%, Avg loss: 3.433949
- value of epoch is 27 Test Error: Accuracy: 37.0%, Avg loss: 2.973505
- value of epoch is 28 Test Error: Accuracy: 30.0%, Avg loss: 3.509221
- value of epoch is 29 Test Error: Accuracy: 29.1%, Avg loss: 3.538610
- value of epoch is 30 Test Error: Accuracy: 54.1%, Avg loss: 2.025637
- value of epoch is 31 Test Error: Accuracy: 55.5%, Avg loss: 1.950219
- value of epoch is 32 Test Error: Accuracy: 56.2%, Avg loss: 1.925000
- value of epoch is 33 Test Error: Accuracy: 54.6%, Avg loss: 2.007267
- value of epoch is 34 Test Error: Accuracy: 55.7%, Avg loss: 1.958895
- value of epoch is 35 Test Error: Accuracy: 54.2%, Avg loss: 2.036361
- value of epoch is 36 Test Error: Accuracy: 52.5%, Avg loss: 2.142057
- value of epoch is 37 Test Error: Accuracy: 52.5%, Avg loss: 2.115632
- value of epoch is 38 Test Error: Accuracy: 52.4%, Avg loss: 2.135156
- value of epoch is 39 Test Error: Accuracy: 54.9%, Avg loss: 2.000643
- value of epoch is 40 Test Error: Accuracy: 49.6%, Avg loss: 2.322968
- value of epoch is 41 Test Error: Accuracy: 53.5%, Avg loss: 2.056279
- value of epoch is 42 Test Error: Accuracy: 54.1%, Avg loss: 2.043148
- value of epoch is 43 Test Error: Accuracy: 55.6%, Avg loss: 1.958567
- value of epoch is 44 Test Error: Accuracy: 54.2%, Avg loss: 2.038740
- value of epoch is 45 Test Error: Accuracy: 52.0%, Avg loss: 2.158267
- value of epoch is 46 Test Error: Accuracy: 54.4%, Avg loss: 2.022273
- value of epoch is 47 Test Error: Accuracy: 53.0%, Avg loss: 2.098935
- value of epoch is 48 Test Error: Accuracy: 55.4%, Avg loss: 1.958429
- value of epoch is 49 Test Error: Accuracy: 54.4%, Avg loss: 2.030314
- value of epoch is 50 Test Error: Accuracy: 53.0%, Avg loss: 2.096478
- value of epoch is 51 Test Error: Accuracy: 55.9%, Avg loss: 1.947644
- value of epoch is 52 Test Error: Accuracy: 54.5%, Avg loss: 2.016503
- value of epoch is 53 Test Error: Accuracy: 53.4%, Avg loss: 2.079320
- value of epoch is 54 Test Error: Accuracy: 54.0%, Avg loss: 2.053968
- value of epoch is 55 Test Error: Accuracy: 56.5%, Avg loss: 1.916731
- value of epoch is 56 Test Error: Accuracy: 55.1%, Avg loss: 1.989408
- value of epoch is 57 Test Error: Accuracy: 54.8%, Avg loss: 2.007595
- value of epoch is 58 Test Error: Accuracy: 54.5%, Avg loss: 2.015584
- value of epoch is 59 Test Error: Accuracy: 56.3%, Avg loss: 1.935730
- value of epoch is 60 Test Error: Accuracy: 63.3%, Avg loss: 1.578893
- value of epoch is 61 Test Error: Accuracy: 64.1%, Avg loss: 1.545642
- value of epoch is 62 Test Error: Accuracy: 64.4%, Avg loss: 1.522229
- value of epoch is 63 Test Error: Accuracy: 64.1%, Avg loss: 1.530667
- value of epoch is 64 Test Error: Accuracy: 64.8%, Avg loss: 1.516458
- value of epoch is 65 Test Error: Accuracy: 64.6%, Avg loss: 1.518174
- value of epoch is 66 Test Error: Accuracy: 65.2%, Avg loss: 1.487593
- value of epoch is 67 Test Error: Accuracy: 65.1%, Avg loss: 1.496522
- value of epoch is 68 Test Error: Accuracy: 65.1%, Avg loss: 1.481308
- value of epoch is 69 Test Error: Accuracy: 65.0%, Avg loss: 1.513329
- value of epoch is 70 Test Error: Accuracy: 65.4%, Avg loss: 1.482963
- value of epoch is 71 Test Error: Accuracy: 65.2%, Avg loss: 1.484999
- value of epoch is 72 Test Error: Accuracy: 65.5%, Avg loss: 1.470484
- value of epoch is 73 Test Error: Accuracy: 65.6%, Avg loss: 1.483683
- value of epoch is 74 Test Error: Accuracy: 65.3%, Avg loss: 1.486447
- value of epoch is 75 Test Error: Accuracy: 65.4%, Avg loss: 1.486100
- value of epoch is 76 Test Error: Accuracy: 65.7%, Avg loss: 1.473895
- value of epoch is 77 Test Error: Accuracy: 65.6%, Avg loss: 1.477824
- value of epoch is 78 Test Error: Accuracy: 65.4%, Avg loss: 1.477430
- value of epoch is 79 Test Error: Accuracy: 65.4%, Avg loss: 1.490247
- value of epoch is 80 Test Error: Accuracy: 65.6%, Avg loss: 1.472184
- value of epoch is 81 Test Error: Accuracy: 65.3%, Avg loss: 1.481460
- value of epoch is 82 Test Error: Accuracy: 66.1%, Avg loss: 1.460380
- value of epoch is 83 Test Error: Accuracy: 64.9%, Avg loss: 1.514267
- value of epoch is 84 Test Error: Accuracy: 65.4%, Avg loss: 1.482626
- value of epoch is 85 Test Error: Accuracy: 65.6%, Avg loss: 1.482990
- value of epoch is 86 Test Error: Accuracy: 65.3%, Avg loss: 1.482573
- value of epoch is 87 Test Error: Accuracy: 65.2%, Avg loss: 1.490366
- value of epoch is 88 Test Error: Accuracy: 65.8%, Avg loss: 1.477603
- value of epoch is 89 Test Error: Accuracy: 65.5%, Avg loss: 1.482135
- value of epoch is 90 Test Error: Accuracy: 66.4%, Avg loss: 1.441268
- value of epoch is 91 Test Error: Accuracy: 66.9%, Avg loss: 1.424808
- value of epoch is 92 Test Error: Accuracy: 66.7%, Avg loss: 1.425573
- value of epoch is 93 Test Error: Accuracy: 67.1%, Avg loss: 1.412974
- value of epoch is 94 Test Error: Accuracy: 66.9%, Avg loss: 1.420117
- value of epoch is 95 Test Error: Accuracy: 67.2%, Avg loss: 1.416592
- value of epoch is 96 Test Error: Accuracy: 67.2%, Avg loss: 1.416405
- value of epoch is 97 Test Error: Accuracy: 66.8%, Avg loss: 1.412191
- value of epoch is 98 Test Error: Accuracy: 67.0%, Avg loss: 1.420147
- value of epoch is 99 Test Error: Accuracy: 66.9%, Avg loss: 1.418949