# denoising_vlg_24
Here is a detailed README file for your project to be uploaded on GitHub:

---

# Denoising Autoencoder for Image Restoration

This project implements a denoising autoencoder using a convolutional neural network (CNN) to remove noise from images. The autoencoder is trained on pairs of noisy and clean images, learning to reconstruct clean images from their noisy counterparts.

## Objective

The objective of this project is to develop a denoising autoencoder capable of effectively removing noise from images. The model is built using convolutional neural networks and is trained on pairs of noisy and clean images.

## Dataset Description

The dataset used for this project consists of two sets of images:
- **Noisy Images:** Images with added noise.
- **Clean Images:** Original clean images without noise.

The images are loaded from specified directories and preprocessed before being used for training and testing the model.

## Data Preprocessing

The images are preprocessed through the following steps:
1. **Loading:** Images are loaded from the specified directories.
2. **Conversion:** Images are converted from BGR to RGB format.
3. **Resizing:** Images are resized to a specified size (128x128).
4. **Normalization:** Image pixel values are normalized to the range [0, 1].

## Data Splitting

The dataset is split into training and testing sets using an 80-20 split. The `train_test_split` function from `sklearn.model_selection` is used to ensure the data is split randomly and consistently.

## Model Architecture

The denoising autoencoder model consists of the following layers:
- **Encoder:**
  - Convolutional layer with 64 filters and ReLU activation
  - MaxPooling layer
  - Convolutional layer with 32 filters and ReLU activation
  - MaxPooling layer
- **Decoder:**
  - Convolutional layer with 32 filters and ReLU activation
  - UpSampling layer
  - Convolutional layer with 64 filters and ReLU activation
  - UpSampling layer
  - Convolutional layer with 3 filters and Sigmoid activation

## Model Training

The model is trained using the following parameters:
- **Optimizer:** Adam
- **Loss Function:** Mean Squared Error (MSE)
- **Epochs:** 50
- **Batch Size:** 32

The model is trained on the training set and validated on the testing set. The training history is recorded for analysis.

## Model Evaluation

The model is evaluated on the testing set using the Mean Squared Error (MSE) loss. Additionally, the Peak Signal-to-Noise Ratio (PSNR) is calculated for each predicted image to measure the quality of the denoised images.

## Results

The model achieves an average PSNR of `average_psnr`, indicating the effectiveness of the denoising process.
Average PSNR: 16.92319678545168

## Installation

To install the required packages, run:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your noisy and clean images in the specified directories.
2. Update the `noisy_folder_path` and `clean_folder_path` variables in the code with the correct paths.
3. Run the script to train the model and evaluate its performance:
```bash
python script.py
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Replace `average_psnr` with the actual average PSNR value calculated from your model evaluation. Also, ensure that the paths and file names in the code and instructions are correctly updated to match your project setup.
