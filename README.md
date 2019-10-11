# <center>A Brief Noise Analysis</center>

<center>Austin Koenig</center>

## Introduction

Describe that this is not a study on machine learning models nor data engineering. This is a study on noise and how it affects the performance of machine learning models through it's prominence in training data. Therefore, the code that was written is relatively hidden from the audience.

### Motivation

Describe briefly the intended audience and why these results are important.

## Data

The data used was generated from a simple sine wave. Noise was added to create training data and testing was performed directly on the function value.

### Noise

A brief discussion on noise in data.

### Training vs. Testing

Discuss the fact that we used lots of noise in the training sets but zero noise in the testing set. Here, we will show the plots of the testing data as well as from a few of the training sets as benchmarks.

## Models

We are comparing the following models to see which will be the least sensitive to noise:
1. Gradient Boost Regressor
2. Deep Neural Network

### What is a Gradient Boost Regressor?

Briefly explain what a gradient boosting is and why we chose this model.

### What is a Deep Neural Network?

Briefly explain what a deep neural network is and why we chose it.

## Results & Discussion

We found that Model 1 fell victim to excess amounts of noise where Model 2 was able to withstand the noise rather well.

Here, we will show the prediction plots and loss plots in order to outline the performance of our model. While looking at a table of error values can be convincing, a picture is worth a thousand words--or numbers, in this case.

We will also highlight a rather contraversial observation of our results.

## Conclusion

Discuss the further implications of these results. If we have relatively clean data, we are more likely to be able to get away with using models which rely on fewer parameters. However, if there is a lot of noise in our data, then a safer solution would be a deep neural network of some kind.# <center>A Brief Noise Analysis</center>

<center>Austin Koenig</center>

## Data

- Domain: $N$ uniformly spaced points in $[-m\pi, m\pi]$ for $m\in\mathbb{Z}$.
- Training Data: Random samples from normal distribution with mean $\sin{(domain)}$ and standard deviation $s^2\in \{ 0.00001, 0.001, 0.1 \}$.
- Testing Data: Exact values of $\sin{(domain)}$.

## Models

- Fully-Connected Deep Neural Network
- Recurrent Neural Network


