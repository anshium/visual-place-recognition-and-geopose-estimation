# Visual Place Recognition and Geopose Estimation

This project focuses on determining the geographic location (latitude and longitude) and camera orientation (angle) from images, a task known as Visual Place Recognition and Geopose Estimation. The experiments are conducted on a dataset of images from the IIIT campus.

## Project Structure

The repository is organized into the following main directories:

-   `swin_transformer/`: Contains scripts for training and validating Swin Transformer models for geopose estimation.
-   `dinov2salad/`: Contains scripts for training and validating DINOv2 models with a SALAD head for geopose estimation.
-   `angle_prediction/`: Contains scripts for training and validating various models for angle prediction.
-   `final_csv_generators/`: Contains scripts to generate the final prediction CSV files.
-   `cleaned_dataset_files/`: Contains the training and validation data labels.

## Geopose Estimation

The primary goal of this task is to predict the latitude and longitude of a given image.

### Models

The following models have been used for this task:

-   **Swin Transformer:** A powerful vision transformer model. (https://arxiv.org/abs/2103.14030)
-   **DINOv2 with SALAD:** A state-of-the-art vision transformer combined with the SALAD technique for visual place recognition. (https://github.com/serizba/salad)

### Training and Validation

-   The models are trained on the IIIT campus dataset, with labels provided in `cleaned_dataset_files/`.
-   The training process involves fine-tuning the pretrained models on the specific task of geopose estimation.
-   Validation is performed using the scripts `swin_validation.py`, `dinov2salad_validation.py`, and `val_and_test_swin_2.py`.

## Angle Prediction

This task focuses on predicting the camera's viewing angle from an image.

### Models

A variety of models have been experimented with for angle prediction:

-   **ConvNext:** A modern convolutional neural network architecture.
-   **DINOv2:** A powerful vision transformer.
-   **EfficientNet:** A family of efficient convolutional neural networks.
-   **Swin Transformer:** A vision transformer model.

### Approaches

Two main approaches have been explored for angle prediction:

1.  **Direct Angle Regression:** Predicting the angle as a single continuous value.
2.  **Sine/Cosine Regression:** Predicting the sine and cosine of the angle to handle the circular nature of angular data.

The code for these experiments can be found in the `angle_prediction/` directory, with subdirectories for each model.

## Dataset

The dataset consists of images from the IIIT campus. The data is split into training and validation sets, with the corresponding labels available in `cleaned_dataset_files/labels_train.csv` and `cleaned_dataset_files/labels_val.csv`.