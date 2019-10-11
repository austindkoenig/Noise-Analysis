# Noise Analysis
# Austin Koenig


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor

import joblib
import os
import shutil
import threading
from dask.distributed import Client

from keras import models
from keras import layers
from keras import optimizers

np.random.seed(12)

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

def check_dirs():
    dirs = ['./files/', './files/figures/', 
            './files/data/', './files/scalers/', 
            './files/models/', './files/models/dnn/', 
            './files/models/rnn/', './files/models/gradboost/',
            './files/figures/']
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

def clean_up():
    shutil.rmtree('./files/')

def generate_data(N = 1000, M = 2, deg_noise = 100):
    # GENERATE DATA
    noise = np.geomspace(10 ** -4, 10 ** 1, deg_noise)
    domain = np.linspace(-M * np.pi, M * np.pi, N)
    tsine = np.sin(domain)
    nsine = np.array([np.random.normal(tsine, n) for n in noise]).T

    data = {
        'tr': {
            'x': np.reshape(domain, (-1, 1)),
            'y': np.reshape(nsine, (-1, nsine.shape[1])),
            'seqx': None,
            'seqy': None
        },

        'ts': {
            'x': np.reshape(domain, (-1, 1)),
            'y': np.reshape(tsine, (-1, 1)),
            'seqx': None,
            'seqy': None
        }
    }

    # convert an array of values into a dataset matrix
    def create_sequences(dataset, lookback = 8, foresight = 4):
        dataX, dataY = [], []
        for i in range(len(dataset) - lookback - foresight):
            obs = dataset[i:(i + lookback)]
            dataX.append(obs)
            dataY.append(dataset[i + lookback + foresight])
        return np.array(dataX), np.array(dataY)
    
    # for k in ['tr', 'ts']:
    #     unscaled_data[k]['seqx'], unscaled_data[k]['seqy'] = create_sequences(unscaled_data[k]['y'])
    #     scaled_data[k]['seqx'], scaled_data[k]['seqy'] = create_sequences(scaled_data[k]['y'])

    joblib.dump(data, './files/data/data')
    return data

def dnn(inShape = (1,)):
    model = models.Sequential([
        layers.Dense(64, activation = 'relu', input_shape = inShape),
        layers.Dense(128, activation = 'relu'),
        layers.Dense(128, activation = 'relu'),
        layers.Dense(128, activation = 'relu'),
        layers.Dense(1, activation = 'tanh')
    ])
    model.compile(loss = 'mse', optimizer = optimizers.Adam())
    return model

# def rnn(inShape = (8,)):
#     model = models.Sequential([
#         layers.GRU(32, activation = 'relu', input_shape = inShape, return_sequences = True),
#         layers.GRU(32, activation = 'relu'),
#         layers.Dense(16, activation = 'relu'),
#         layers.Dense(1, activation = 'linear')
#     ])
#     model.compile(loss = 'mse', optimizer = optimizers.Adam())
#     return model

def evaluation():
    data = generate_data()

    gberrs = []
    dnnerrs = []

    def fit_eval_grad_boost(noise):
        boost = GradientBoostingRegressor(loss = 'ls', learning_rate = 0.1, n_estimators = 1000, max_depth = 5)
        boost.fit(data['tr']['x'], data['tr']['y'][:, noise])
        test_err = np.mean((data['ts']['y'] - boost.predict(data['ts']['x'])) ** 2)
        joblib.dump(boost, './files/models/gradboost/model')
        joblib.dump(test_err, './files/models/gradboost/test_err')
        # print(f"    Testing GradBoost.\n    Degree of Noise: {noise}")
        # print(f"        Test Error: {test_err}")
        return boost, test_err

    def fit_eval_dnn(noise):
        deep = dnn()
        deephist = deep.fit(data['tr']['x'], 
                            data['tr']['y'][:, noise], 
                            batch_size = 16, 
                            epochs = 20,
                            validation_split = 0.3,
                            verbose = 0)
        deepeval = deep.evaluate(data['ts']['x'], data['ts']['y'])
        # print(f"    Testing DeepNN.\n    Degree of Noise: {noise}")
        # print(f"        Test Error: {deepeval}")
        joblib.dump(deep, './files/models/dnn/model')
        joblib.dump(deephist, './files/models/dnn/history')
        joblib.dump(deepeval, './files/models/dnn/evaluation')
        return deep, deephist, deepeval
    
    # def fit_eval_rnn():
    #     recurr = rnn()
    #     recurrhist = recurr.fit(data['tr']['seqx'],
    #                             data['tr']['seqy'],
    #                             batch_size = 1, 
    #                             epochs = 20,
    #                             validation_data = (data['val']['seqx'], data['val']['seqy']),
    #                             verbose = 2)
    #     recurreval = recurr.evaluate(data['ts']['seqx'], data['ts']['seqy'])
    #     joblib.dump(recurr, './files/models/rnn/model')
    #     joblib.dump(recurrhist, './files/models/rnn/history')
    #     joblib.dump(recurreval, './files/models/rnn/evaluation')
    #     return recurr, recurrhist, recurreval

    # joblib.Parallel(n_jobs = 3)(joblib.delayed(fit_eval_dnn)(i) for i in range(data['tr']['y'].shape[1]))
    # joblib.Parallel(n_jobs = 3)(joblib.delayed(fit_eval_grad_boost)(i) for i in range(data['tr']['y'].shape[1]))

    #np.apply_along_axis(fit_eval_grad_boost, 0, np.arange(data['tr']['y'].shape[1]))
    #np.apply_along_axis(fit_eval_dnn, 0, range(3))

    for i in range(data['tr']['y'].shape[1]):
        m1, e1 = fit_eval_grad_boost(i)
        m2, h2, e2 = fit_eval_dnn(i)
        gberrs.append(e1)
        dnnerrs.append(e2)

    mnames = ['GBR', 'DNN']

    print(f"\nModel: {mnames[0]}")
    print("    | Deg. Noise | Test Error |")
    for i in range(len(gberrs)):
        print(f"    | {i:10} | {round(gberrs[i], 4):10} |")

    print(f"\nModel: {mnames[1]}")
    print("    | Deg. Noise | Test Error |")
    for i in range(len(dnnerrs)):
        print(f"    | {i:10} | {round(dnnerrs[i], 4):10} |")

    # with joblib.parallel_backend('dask'):
    #     joblib.Parallel(verbose = 100)(joblib.delayed(fit_eval_dnn)(i) for i in range(3))

    # VISUALIZE DATA
    loss_figure = plt.figure(figsize = (20, 15))
    aa = loss_figure.add_subplot(1, 1, 1)
    aa.plot(np.array(gberrs), label = "GradBoost Loss")
    aa.plot(np.array(dnnerrs), label = "DeepNN Loss")
    aa.set_xlabel("Degree of Noise")
    aa.set_ylabel("Loss")
    aa.title.set_text("Losses")
    aa.legend()
    loss_figure.savefig('./files/figures/loss_figure.pdf')

    test_figure = plt.figure(figsize = (20, 15))
    a1 = test_figure.add_subplot(1, 1, 1)
    a1.plot(data['ts']['x'], data['ts']['y'], 'k-', label = "True Sine")
    a1.title.set_text("Test Set")
    a1.legend()
    test_figure.savefig('./files/figures/test_figure.pdf')

    NN = data['tr']['y'].shape[1]
    for i in range(NN):
        if (i % 10 == 0):
            train_figure = plt.figure(figsize = (20, 15))
            ax = train_figure.add_subplot(1, 1, 1)
            ax.plot(data['tr']['x'], data['tr']['y'][:, i], 'ko', label = f"Deg. Noise: {i}", markersize = 2)
            ax.title.set_text(f"Training Set {i}")
            ax.legend()
            train_figure.savefig(f'./files/figures/train_figure_n{i}.pdf')
    #plt.show()

if __name__ == "__main__":
    print("----NOISE STUDY----")
    clean_up()
    #client = Client(processes = False)
    clean = False
    print("Creating directories...")
    check_dirs()
    print("Directories created.")
    print("Creating and evaluating models...")
    evaluation()
    print("Models created and evaluated.")
    if clean: 
        print("Cleaning up...")
        clean_up()
        print("All cleaned up.")
