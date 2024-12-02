# GAN-Data-Generation-Project
GAN Data Generation Project
Overview
This project implements a Generative Adversarial Network (GAN) to generate synthetic tabular data. It uses PyTorch to create both a Generator and Discriminator network that work together to produce high-quality synthetic data samples that match the statistical properties of the original dataset.
Features

Implementation of GAN architecture using PyTorch
Generator network to create synthetic data samples
Discriminator network to distinguish between real and fake data
Data preprocessing including standardization
Training monitoring with loss visualization
Synthetic data generation and export capabilities

Requirements

Python 3.x
PyTorch
Pandas
NumPy
Matplotlib
Scikit-learn

Project Structure
The project consists of several key components:

Data preprocessing and loading
GAN architecture implementation
Training loop
Visualization tools
Synthetic data generation

Usage

Load and preprocess the data:
df = pd.read_csv('gan_dataset.csv')
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.values)

Define and initialize the GAN models:
generator = Generator(latent_dim, output_dim)
discriminator = Discriminator(input_dim)

Train the GAN:
for epoch in range(num_epochs):
    # Training loop implementation

Generate synthetic samples:
z = torch.randn(100, latent_dim)
generated_data = generator(z)

Model Architecture

Generator: Multi-layer neural network that transforms random noise into synthetic data samples
Discriminator: Binary classifier that distinguishes between real and generated samples

Training Process
The training process alternates between:

Training the discriminator on real and fake data
Training the generator to produce more convincing fake data
Monitoring loss values for both networks

Results

Generated synthetic data matches statistical properties of original dataset
Loss curves show convergence of both networks
Quality metrics show good similarity between real and synthetic data