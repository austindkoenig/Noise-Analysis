'''
Noise Analysis
Visual Storytelling
Austin Koenig
'''

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
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

        if out_file:
            sys.stdout = open(os.path.join(self.directories['root'], 'log.md'), 'w')
    
    def check_dirs(self):
        for d in self.directories:
            if not os.path.exists(self.directories[d]):
                os.makedirs(self.directories[d])

    def clean_up(self):
        if os.path.exists(self.directories['root']):
            shutil.rmtree(self.directories['root'])

    def generate_data(self, N = 1000, M = 2, deg_noise = 10, lower = 10 ** -3, upper = 1):
        geom_noise = np.geomspace(lower, upper, deg_noise)
        linear_noise = np.linspace(lower, upper, deg_noise)
        log_noise = np.logspace(lower, upper, deg_noise)
        domain = np.linspace(-M * np.pi, M * np.pi, N)
        tsine = np.sin(domain)
        nsine = np.array([np.random.normal(tsine, n) for n in linear_noise]).T

        data = {
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
        noise_axis.plot(log_noise, label = "Logarithmically Spaced")
        noise_axis.title.set_text("Noise Levels")
        noise_axis.set_xlabel("Degree of Noise")
        noise_axis.set_ylabel("Noise Level")
        noise_axis.legend()
        noise_figure.savefig(os.path.join(self.directories['figures'], 'noise.pdf'))

        joblib.dump(data, os.path.join(self.directories['data'], 'data'))
        self.checkpoints['data'] = True
    
    def generate_models(self):
        self.dnn()
        self.gbr()
        self.pnr()
    
    def dnn(self, key = ''):
        IN = layers.Input(shape = (1,))
        y = layers.Dense(1024, activation = 'relu')(IN)
        y = layers.Dense(1024, activation = 'relu')(y)
        y = layers.Dense(1024, activation = 'relu')(y)
        OUT = layers.Dense(1, activation = 'tanh')(y)
        self.regressors[key] = models.Model(inputs = IN, outputs = OUT)
        self.regressors[key].compile(loss = 'mse', optimizer = optimizers.Adam())
        print("\nDeep Neural Network")
        self.regressors[key].summary()
    
    def gbr(self, key = '', ests = 1000, depth = 10):
        self.regressors[key] = GradientBoostingRegressor(loss = 'ls', 
                                                         learning_rate = 0.1, 
                                                         n_estimators = ests, 
                                                         max_depth = depth, 
                                                         validation_fraction = 0.3, 
                                                         verbose = 1)
        print("\nGradient Boost Regressor")
        print(json.dumps(self.regressors[key].get_params(), indent = 2))
    
    def pnr(self, key = '', degree = 6):
        self.regressors[key] = LinearRegression()
        print("\nPolynomial Regressor")
        print(json.dumps(self.regressors[key].get_params(), indent = 2))
    
    def evaluation(self, EPCHS = 500, BATCH = 64):
        dnn_key = 'DNN'
        gbr_key = 'GBR'
        pnr_key = 'PNR'

        self.predictions[dnn_key] = []
        self.predictions[gbr_key] = []
        self.predictions[pnr_key] = []
        self.tests[dnn_key] = []
        self.tests[gbr_key] = []
        self.tests[pnr_key] = []

        # fit models
        for i in range(self.data['train']['y'].shape[1]):
            self.dnn(dnn_key)
            self.regressors[dnn_key].fit(self.data['train']['x'], self.data['train']['y'][:, i],
                                         batch_size = BATCH, epochs = EPCHS,
                                         validation_split = 0.3, verbose = 0)
            self.tests[dnn_key].append(self.regressors[dnn_key].evaluate(self.data['test']['x'], self.data['test']['y']))

            self.gbr(gbr_key)
            self.regressors[gbr_key].fit(self.data['train']['x'], self.data['train']['y'][:, i])
            self.tests[gbr_key] = self.regressors[gbr_key].evaluate(self.data['test']['x'], self.data['test'])

            self.pnr(pnr_key)
            x_poly = PolynomialFeatures(degree = 10).fit_transform(self.data['train']['x'])
            self.regressors[pnr_key].fit(x_poly, self.data['train']['y'][:, i])

if __name__ == '__main__':
    na = NoiseAnalysis()
    na.clean_up()
    na.check_dirs()
    na.generate_data()
    na.evaluation()