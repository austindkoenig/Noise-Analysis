# Noise Analysis
# Austin Koenig


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import joblib
import os
import shutil

np.random.seed(12)

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

def check_dirs():
    dirs = ['./files/', './files/figures/', './files/data/', './files/scalers/', './files/models/']
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
            'y': nsine[:tr_split, :]
        },

        'val': {
            'x': domain[tr_split:(tr_split + val_split)],
            'y': nsine[tr_split:(tr_split + val_split), :]
        }, 

        'ts': {
            'x': domain[(tr_split + val_split):],
            'y': nsine[(tr_split + val_split):, :]
        }
    }

    # SCALE DATA
    scaler = MinMaxScaler()
    scaled_data = {
        'tr': {
            'x': unscaled_data['tr']['x'],
            'y': scaler.fit_transform(unscaled_data['tr']['y'])
        },

        'val': {
            'x': unscaled_data['val']['x'],
            'y': scaler.transform(unscaled_data['val']['y'])
        }, 

        'ts': {
            'x': unscaled_data['ts']['x'],
            'y': scaler.transform(unscaled_data['ts']['y'])
        }
    }
    
    joblib.dump(splits, './files/data/splits')
    joblib.dump(unscaled_data, './files/data/unscaled_data')
    joblib.dump(scaled_data, './files/data/scaled_data')
    joblib.dump(scaler, './files/scalers/scaler')

    return unscaled_data, scaled_data, scaler

def poly_interp():
    pass

def conv1d():
    pass

def rnn():
    pass

def evaluation():
    pass


if __name__ == "__main__":
    print("----NOISE STUDY----")
    clean = True
    print("Creating directories...")
    check_dirs()
    print("Directories created.")
    print("Generating data...")
    generate_data()
    print("Data generated.")
    if clean: 
        print("Cleaning up...")
        clean_up()
        print("All cleaned up.")
