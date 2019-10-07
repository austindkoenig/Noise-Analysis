# <center>A Brief Noise Analysis</center>

<center>Austin Koenig</center>

## Data

- Domain: $N$ uniformly spaced points in $[-m\pi, m\pi]$ for $m\in\mathbb{Z}$.
- Training Data: Random samples from normal distribution with mean $\sin{(domain)}$ and standard deviation $s^2\in \{ 0.00001, 0.001, 0.1 \}$.
- Testing Data: Exact values of $\sin{(domain)}$.

## Models

- Polynomial Interpolation
- 1D Convolutional Neural Network
- Recurrent Neural Network
