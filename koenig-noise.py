# Noise Analysis
# Austin Koenig


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import joblib
import os
import shutil
import threading

from keras import models
from keras import layers
from keras import optimizers

np.random.seed(12)

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

def check_dirs():
    dirs = ['./files/', './files/figures/', './files/data/', './files/scalers/', './files/models/', './files/models/dnn/', './files/models/rnn/']
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

def clean_up():
    shutil.rmtree('./files/')

def generate_data(N = 1000, M = 2, deg_noise = 3):
    # GENERATE DATA
    noise = np.geomspace(10 ** -3, 10 ** -1, deg_noise)
    domain = np.linspace(-M * np.pi, M * np.pi, N)
    tsine = np.sin(domain)
    nsine = np.array([np.random.normal(tsine, n) for n in noise]).T

    # VISUALIZE DATA
    # figure = plt.figure(figsize = (20, 20))
    # a1 = figure.add_subplot(len(noise) + 1, 1, 1)
    # a1.plot(domain, tsine, 'k-', label = "True Sine")
    # a1.title.set_text("Test Set")
    # a1.legend()

    # if not os.path.exists('./figures/'):
    #     os.mkdir('./figures/')
    # #test_figure.savefig('./figures/test_figure.pdf')

    # #train_figure = plt.figure(figsize = (20, 20))
    
    # axes = []
    # for i in range(1, len(nsine) + 1):
    #     axes.append(figure.add_subplot(len(noise), 1, i + 1))
    #     axes[i - 1].plot(domain, nsine[:, i - 1], 'ko', label = f"Noise: {noise[i - 1]}", markersize = 2)
    #     axes[i - 1].title.set_text(f"Training Set {i}")
    #     axes[i - 1].legend()
    
    # #train_figure.savefig('./figures/train_figure.pdf')
    # plt.show()

    # SPLIT DATA
    tr_rat = 0.6
    val_rat = 0.2
    ts_rat = 0.2
    tr_split = round(tr_rat * tsine.shape[0])
    val_split = round(val_rat * tsine.shape[0])
    ts_split = round(ts_rat * tsine.shape[0])

    splits = {
        'train': (tr_split, tr_rat),
        'validation': (val_split, val_rat),
        'test': (ts_split, ts_rat)
    }

    unscaled_data = {
        'tr': {
            'x': domain[:tr_split],
            'y': nsine[:tr_split, :],
            'seqx': None,
            'seqy': None
        },

        'val': {
            'x': domain[tr_split:(tr_split + val_split)],
            'y': nsine[tr_split:(tr_split + val_split), :],
            'seqx': None,
            'seqy': None
        }, 

        'ts': {
            'x': domain[(tr_split + val_split):],
            'y': nsine[(tr_split + val_split):, :],
            'seqx': None,
            'seqy': None
        }
    }

    # SCALE DATA
    scaler = MinMaxScaler()
    scaled_data = {
        'tr': {
            'x': unscaled_data['tr']['x'],
            'y': scaler.fit_transform(unscaled_data['tr']['y']),
            'seqx': None,
            'seqy': None
        },

        'val': {
            'x': unscaled_data['val']['x'],
            'y': scaler.transform(unscaled_data['val']['y']),
            'seqx': None,
            'seqy': None
        }, 

        'ts': {
            'x': unscaled_data['ts']['x'],
            'y': scaler.transform(unscaled_data['ts']['y']),
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
    
    for k in ['tr', 'val', 'ts']:
        unscaled_data[k]['seqx'], unscaled_data[k]['seqy'] = create_sequences(unscaled_data[k]['y'][:, 0])
        scaled_data[k]['seqx'], scaled_data[k]['seqy'] = create_sequences(scaled_data[k]['y'][:, 0])
    
    joblib.dump(splits, './files/data/splits')
    joblib.dump(unscaled_data, './files/data/unscaled_data')
    joblib.dump(scaled_data, './files/data/scaled_data')
    joblib.dump(scaler, './files/scalers/scaler')
    return unscaled_data, scaled_data, scaler

def dnn(inShape = (1,)):
    model = models.Sequential([
        layers.Dense(32, activation = 'relu', input_shape = inShape),
        layers.Dense(64, activation = 'relu'),
        layers.Dense(16, activation = 'relu'),
        layers.Dense(1, activation = 'tanh')
    ])
    model.compile(loss = 'mae', optimizer = optimizers.Adam())
    return model

def rnn(inShape = (8,)):
    model = models.Sequential([
        layers.GRU(32, activation = 'relu', input_shape = inShape, return_sequences = True),
        layers.GRU(32, activation = 'relu'),
        layers.Dense(16, activation = 'relu'),
        layers.Dense(1, activation = 'linear')
    ])
    model.compile(loss = 'mae', optimizer = optimizers.Adam())
    return model

def evaluation():
    raw_data, data, scaler = generate_data()

    def fit_eval_dnn(noise):
        deep = dnn()
        deephist = deep.fit(data['tr']['x'], 
                            data['tr']['y'][:, noise], 
                            batch_size = 1, 
                            epochs = 20,
                            validation_data = (data['val']['x'], data['val']['y'][:, noise]),
                            verbose = 2)
        deepeval = deep.evaluate(data['ts']['x'], data['ts']['y'][:, noise])
        joblib.dump(deep, './files/models/dnn/model')
        joblib.dump(deephist, './files/models/dnn/history')
        joblib.dump(deepeval, './files/models/dnn/evaluation')
        return deep, deephist, deepeval
    
    def fit_eval_rnn():
        recurr = rnn()
        recurrhist = recurr.fit(data['tr']['seqx'],
                                data['tr']['seqy'],
                                batch_size = 1, 
                                epochs = 20,
                                validation_data = (data['val']['seqx'], data['val']['seqy']),
                                verbose = 2)
        recurreval = recurr.evaluate(data['ts']['seqx'], data['ts']['seqy'])
        joblib.dump(recurr, './files/models/rnn/model')
        joblib.dump(recurrhist, './files/models/rnn/history')
        joblib.dump(recurreval, './files/models/rnn/evaluation')
        return recurr, recurrhist, recurreval
    
    # thread1 = threading.Thread(target = fit_eval_dnn, name = 'DNN')
    # thread2 = threading.Thread(target = fit_eval_rnn, name = 'RNN')
    # thread1.start()
    # thread2.start()
    # thread1.join()
    # thread2.join()

    return joblib.Parallel(n_jobs = 3)(joblib.delayed(fit_eval_dnn)(i) for i in range(data['tr']['y'].shape[1]))

if __name__ == "__main__":
    print("----NOISE STUDY----")
    clean = False
    print("Creating directories...")
    check_dirs()
    print("Directories created.")
    print("Generating data...")
    generate_data()
    print("Data generated.")
    print("Creating and evaluating models...")
    evaluation()
    print("Models created and evaluated.")
    if clean: 
        print("Cleaning up...")
        clean_up()
        print("All cleaned up.")
