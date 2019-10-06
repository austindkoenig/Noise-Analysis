# Noise Study
# Austin Koenig

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12)


def prepare_data(N = 500, 
                domain = (-2 * np.pi, 2 * np.pi),
                noise = [0.0001, 0.01, 0.1]):
    x = np.linspace(domain[0], domain[1], N)
    rawy = np.sin(x)
    y = [np.random.normal(rawy, n) for n in noise]
    plt.plot(x, rawy, '-', label = "True Sine")
    for i in range(len(y)):
        plt.plot(x, y[i], '.', label = f"Noise {noise[i]}", markersize = 3)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Superposition of Noisy Sine onto Sine")
    plt.legend()
    plt.show()

def poly_interp():
    pass

def conv1d():
    pass

def rnn():
    pass

def evaluation(model):
    pass


if __name__ == "__main__":
    print("----NOISE STUDY----")
    prepare_data()
