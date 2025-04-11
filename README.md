This repository contains the implementation of a project exploring dimensionality reduction using autoencoders, with a comparison to traditional methods like Principal Component Analysis (PCA). The project uses the MNIST dataset to demonstrate how autoencoders can effectively compress high-dimensional data into a lower-dimensional latent space while preserving essential features, outperforming PCA in reconstruction quality and downstream tasks.

The project is based on the report "Dimensionality Reduction Using Autoencoders" and includes code, results, and documentation to replicate the experiments.
Table of Contents
Project Overview
Dimensionality reduction is crucial in machine learning to tackle the curse of dimensionality, enabling efficient computation, better visualization, and improved model performance. This project focuses on autoencoders, neural networks that learn compressed data representations in an unsupervised manner. Unlike linear methods like PCA, autoencoders capture non-linear patterns, making them ideal for complex datasets like images.

Objectives
Train an autoencoder to reduce the dimensionality of MNIST images (784D to 32D).
Compare its performance with PCA in terms of reconstruction error and classification accuracy on a downstream task.
Highlight the strengths and weaknesses of both methods.
Key Findings
Autoencoders achieved a lower reconstruction error (MSE ≈ 0.015) compared to PCA (MSE ≈ 0.025).
Latent representations from autoencoders led to higher classification accuracy (95.2%) than PCA (92.1%).
Autoencoders excel in capturing non-linear patterns but require more computational resources.
Methodology
The project implements a fully connected autoencoder with the following components:

Encoder: Compresses input data (e.g., 784D MNIST images) into a lower-dimensional latent space (32D) using layers like 784 → 256 → 128 → 64 → 32 with ReLU activations.
Decoder: Reconstructs the original data from the latent space using a symmetric architecture (32 → 64 → 128 → 256 → 784) with a sigmoid output.
Training: Uses the Adam optimizer and Mean Squared Error (MSE) loss, trained for 50 epochs on MNIST with early stopping to prevent overfitting.
Baseline: PCA reduces the data to 32 dimensions for comparison.
Evaluation:
Reconstruction quality via MSE.
Visual inspection of reconstructed images.
Classification accuracy using logistic regression on the 32D representations.
The MNIST dataset (60,000 training and 10,000 test images) is preprocessed by normalizing pixel values to [0, 1] and flattening images into 784D vectors.

Results
Reconstruction Quality:
Autoencoder: Clear, sharp reconstructions with MSE ≈ 0.015.
PCA: Blurrier images with MSE ≈ 0.025, losing fine details.
Downstream Task:
Autoencoder latent features: 95.2% accuracy in digit classification.
PCA features: 92.1% accuracy, less discriminative due to linear constraints.
Visualizations:
Reconstructed MNIST digits (see results/ folder for sample images).
Latent space analysis shows better clustering with autoencoders.
For detailed results, refer to the report in docs/report.pdf.

Installation
To set up the project locally, follow these steps:

Clone the repository:
bash

Collapse

Wrap

Copy
git clone https://github.com/your-username/dimensionality-reduction-autoencoders.git
cd dimensionality-reduction-autoencoders
Create a virtual environment (optional but recommended):
bash

Collapse

Wrap

Copy
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:
bash

Collapse

Wrap

Copy
pip install -r requirements.txt
Usage
Prepare the data:
The MNIST dataset is automatically downloaded via TensorFlow/PyTorch when running the code.
Train the autoencoder:
bash

Collapse

Wrap

Copy
python train_autoencoder.py
This script trains the model and saves the weights in models/.
Compare with PCA:
bash

Collapse

Wrap

Copy
python compare_pca.py
This evaluates both methods and generates plots in results/.
Visualize results:
Check reconstructed images and latent space plots in results/.
Run visualize.py for interactive visualizations:
bash

Collapse

Wrap

Copy
python visualize.py
Run downstream classification:
bash

Collapse

Wrap

Copy
python classify.py
This tests the latent representations with logistic regression.
