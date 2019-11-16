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
        
        # scale the errors
        gbr_scaler = StandardScaler()
        pnr_scaler = StandardScaler()
        ann_scaler = StandardScaler()
        dnn_scaler = StandardScaler()

        self.tests[gbr_key] = gbr_scaler.fit_transform(np.array(self.tests[gbr_key]).reshape((-1, 1)))
        self.tests[pnr_key] = pnr_scaler.fit_transform(np.array(self.tests[pnr_key]).reshape((-1, 1)))
        self.tests[ann_key] = ann_scaler.fit_transform(np.array(self.tests[ann_key]).reshape((-1, 1)))
        self.tests[dnn_key] = dnn_scaler.fit_transform(np.array(self.tests[dnn_key]).reshape((-1, 1)))

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
        error_axis.legend()
        error_figure.savefig(os.path.join(self.directories['figures'], 'errors.pdf'))

        for i in range(self.data['train']['y'].shape[1]):
            # every third degree of noise...
            if i % 4 == 0:
                prediction_figure = plt.figure(figsize = (20, 15))
                prediction_axis = prediction_figure.add_subplot(111)
                prediction_axis.plot(self.data['test']['x'], self.predictions[gbr_key][i], label = "Gradient Boost Regression")
                prediction_axis.plot(self.data['test']['x'], self.predictions[pnr_key][i], label = "Polynomial Regression")
                prediction_axis.plot(self.data['test']['x'], self.predictions[ann_key][i], label = "Artificial Neural Network")
                prediction_axis.plot(self.data['test']['x'], self.predictions[dnn_key][i], label = "Deep Neural Network")
                prediction_axis.title.set_text(f"Model Predictions of {i}th Degree of Noise")
                prediction_axis.set_xlabel("x")
                prediction_axis.set_ylabel("y")
                prediction_axis.legend()
                prediction_figure.savefig(os.path.join(self.directories['figures'], f'prediction--noise-{i}.pdf'))

if __name__ == '__main__':
    na = NoiseAnalysis()
    na.generate_data()
    na.evaluation()
