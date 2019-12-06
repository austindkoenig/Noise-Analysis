'''
Noise Analysis
Visual Storytelling
Austin Koenig
'''

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

import joblib
import json
import os
import sys
import shutil

from keras import models
from keras import layers
from keras import optimizers

np.random.seed(12)
plt.rcParams['image.cmap'] = 'Dark2'
plt.rcParams['axes.prop_cycle'] = plt.cycler(color = plt.cm.Dark2.colors)

class NoiseAnalysis(object):
    def __init__(self, out_file = True):
        self.checkpoints = {
            'data': False,
            'evaluation': False,
            'figures': False
        }

        self.directories = {
            'root': './results/',
            'data': './results/data/',
            'figures': './results/figures/',
            'metrics': './results/metrics/'
        }

        self.data = {}
        self.regressors = {}
        self.predictions = {}
        self.tests = {}
        self.standardized_tests = {}

        self.clean_up()
        self.check_dirs()

        if out_file:
            sys.stdout = open(os.path.join(self.directories['root'], 'log.md'), 'w')

    def check_dirs(self):
        for d in self.directories:
            if not os.path.exists(self.directories[d]):
                os.makedirs(self.directories[d])

    def clean_up(self):
        if os.path.exists(self.directories['root']):
            shutil.rmtree(self.directories['root'])

    def generate_data(self, N = 1000, M = 2, deg_noise = 16, lower = 10 ** -3, upper = 1):
        geom_noise = np.geomspace(lower, upper, deg_noise)
        linear_noise = np.linspace(lower, upper, deg_noise)
        domain = np.linspace(-M * np.pi, M * np.pi, N)
        tsine = np.sin(domain)
        nsine = np.array([np.random.normal(tsine, n) for n in linear_noise]).T

        self.data = {
            'train': {
                'x': np.reshape(domain, (-1, 1)),
                'y': np.reshape(nsine, (-1, nsine.shape[1]))
            },

            'test': {
                'x': np.reshape(domain, (-1, 1)),
                'y': np.reshape(tsine, (-1, 1))
            }
        }

        # generate plot of noise levels
        noise_figure = plt.figure(figsize = (20, 15))
        noise_axis = noise_figure.add_subplot(111)
        noise_axis.plot(geom_noise, label = "Geometrically Spaced")
        noise_axis.plot(linear_noise, label = "Linearly Spaced")
        noise_axis.title.set_text("Noise Levels")
        noise_axis.set_xlabel("Degree of Noise")
        noise_axis.set_ylabel("Noise Level")
        noise_axis.legend()

        noise_axis.spines['top'].set_visible(False)
        noise_axis.spines['bottom'].set_visible(False)
        noise_axis.spines['left'].set_visible(False)
        noise_axis.spines['right'].set_visible(False)
        noise_axis.tick_params(length = 0.0, width = 0.0)

        noise_figure.savefig(os.path.join(self.directories['figures'], 'noise.pdf'))

        joblib.dump(self.data, os.path.join(self.directories['data'], 'data'))
        self.checkpoints['data'] = True

    def ann(self):
        IN = layers.Input(shape = (1,))
        y = layers.Dense(1024, activation = 'relu')(IN)
        OUT = layers.Dense(1, activation = 'tanh')(y)
        m = models.Model(inputs = IN, outputs = OUT)
        m.compile(loss = 'mse', optimizer = optimizers.Adam())
        print("\nShallow Neural Network")
        m.summary()
        return m
    
    def dnn(self):
        IN = layers.Input(shape = (1,))
        y = layers.Dense(1024, activation = 'relu')(IN)
        y = layers.Dense(1024, activation = 'relu')(y)
        y = layers.Dense(1024, activation = 'relu')(y)
        OUT = layers.Dense(1, activation = 'tanh')(y)
        m = models.Model(inputs = IN, outputs = OUT)
        m.compile(loss = 'mse', optimizer = optimizers.Adam())
        print("\nDeep Neural Network")
        m.summary()
        return m
    
    def gbr(self):
        m = GradientBoostingRegressor(loss = 'ls', 
                                      learning_rate = 0.1, 
                                      n_estimators = 100, 
                                      max_depth = 5, 
                                      validation_fraction = 0.3, 
                                      verbose = 1)
        print("\nGradient Boost Regressor")
        print(json.dumps(m.get_params(), indent = 2))
        return m
    
    def pnr(self):
        m = LinearRegression()
        print("\nPolynomial Regressor")
        print(json.dumps(m.get_params(), indent = 2))
        return m
    
    def evaluation(self, EPCHS = 100, BATCH = 64):
        ann_key = 'ANN'
        dnn_key = 'DNN'
        gbr_key = 'GBR'
        pnr_key = 'PNR'

        self.regressors[dnn_key] = []
        self.regressors[gbr_key] = []
        self.regressors[pnr_key] = []
        self.regressors[ann_key] = []
        self.predictions[dnn_key] = []
        self.predictions[gbr_key] = []
        self.predictions[pnr_key] = []
        self.predictions[ann_key] = []
        self.tests[dnn_key] = []
        self.tests[gbr_key] = []
        self.tests[pnr_key] = []
        self.tests[ann_key] = []

        # fit models
        for i in range(self.data['train']['y'].shape[1]):
            self.regressors[ann_key].append(self.ann())
            self.regressors[ann_key][i].fit(self.data['train']['x'], self.data['train']['y'][:, i],
                                         batch_size = BATCH, epochs = EPCHS,
                                         validation_split = 0.3, verbose = 0)
            self.predictions[ann_key].append(self.regressors[ann_key][i].predict(self.data['test']['x']))
            self.tests[ann_key].append(np.mean((self.predictions[ann_key][i] - self.data['test']['y']) ** 2))

            self.regressors[dnn_key].append(self.dnn())
            self.regressors[dnn_key][i].fit(self.data['train']['x'], self.data['train']['y'][:, i],
                                         batch_size = BATCH, epochs = EPCHS,
                                         validation_split = 0.3, verbose = 0)
            self.predictions[dnn_key].append(self.regressors[dnn_key][i].predict(self.data['test']['x']))
            self.tests[dnn_key].append(np.mean((self.predictions[dnn_key][i] - self.data['test']['y']) ** 2))

            self.regressors[gbr_key].append(self.gbr())
            self.regressors[gbr_key][i].fit(self.data['train']['x'], self.data['train']['y'][:, i])
            self.predictions[gbr_key].append(self.regressors[gbr_key][i].predict(self.data['test']['x']))
            self.tests[gbr_key].append(np.mean((self.predictions[gbr_key][i] - self.data['test']['y']) ** 2))

            self.regressors[pnr_key].append(self.pnr())
            fitter = PolynomialFeatures(degree = 10)
            x_poly = fitter.fit_transform(self.data['train']['x'])
            self.regressors[pnr_key][i].fit(x_poly, self.data['train']['y'][:, i])
            self.predictions[pnr_key].append(self.regressors[pnr_key][i].predict(fitter.transform(self.data['test']['x'])))
            self.tests[pnr_key].append(np.mean((self.predictions[pnr_key][i] - self.data['test']['y']) ** 2))


        # plots
        error_figure = plt.figure(figsize = (20, 15))
        error_axis = error_figure.add_subplot(111)
        error_axis.plot(self.tests[gbr_key], label = "Gradient Boost Regression")
        error_axis.plot(self.tests[pnr_key], label = "Polynomial Regression")
        error_axis.plot(self.tests[ann_key], label = "Artificial Neural Network")
        error_axis.plot(self.tests[dnn_key], label = "Deep Neural Network")
        error_axis.title.set_text("Model Errors")
        error_axis.set_xlabel("Degree of Noise")
        error_axis.set_ylabel("Error")
        error_axis.spines['top'].set_visible(False)
        error_axis.spines['bottom'].set_visible(False)
        error_axis.spines['left'].set_visible(False)
        error_axis.spines['right'].set_visible(False)
        error_axis.tick_params(length = 0.0, width = 0.0)
        error_axis.legend()

        error_figure.savefig(os.path.join(self.directories['figures'], 'errors.pdf'))


        prediction_figure, prediction_axes = plt.subplots(nrows = 2, ncols = 2, sharex = 'all', sharey = 'all', figsize = (20, 15))
        prediction_noises = [0, 6, 12, 15]
        k = 0

        for i in [0, 1]:
            for j in [0, 1]:
                prediction_axes[i, j].plot(self.data['test']['x'], self.predictions[gbr_key][prediction_noises[k]], label = "Gradient Boost Regression")
                prediction_axes[i, j].plot(self.data['test']['x'], self.predictions[pnr_key][prediction_noises[k]], label = "Polynomial Regression")
                prediction_axes[i, j].plot(self.data['test']['x'], self.predictions[ann_key][prediction_noises[k]], label = "Artificial Neural Network")
                prediction_axes[i, j].plot(self.data['test']['x'], self.predictions[dnn_key][prediction_noises[k]], label = "Deep Neural Network")

                prediction_axes[i, j].set_title(f"Model Predictions of {prediction_noises[k]}th Degree of Noise")
                prediction_axes[i, j].spines['top'].set_visible(False)
                prediction_axes[i, j].spines['bottom'].set_visible(False)
                prediction_axes[i, j].spines['left'].set_visible(False)
                prediction_axes[i, j].spines['right'].set_visible(False)
                prediction_axes[i, j].tick_params(length = 0.0, width = 0.0)

                if k == 3:
                    handles, labels = prediction_axes[i, j].get_legend_handles_labels()
                    prediction_figure.legend(handles, labels, loc = 'lower center')

                k += 1
        
        prediction_figure.savefig(os.path.join(self.directories['figures'], f'predictions.pdf'), transparent = True)

if __name__ == '__main__':
    na = NoiseAnalysis()
    na.generate_data()
    na.evaluation()
