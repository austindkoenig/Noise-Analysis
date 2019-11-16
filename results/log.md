
Shallow Neural Network
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 1)                 0         
_________________________________________________________________
dense_1 (Dense)              (None, 1024)              2048      
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 1025      
=================================================================
Total params: 3,073
Trainable params: 3,073
Non-trainable params: 0
_________________________________________________________________

Deep Neural Network
Model: "model_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         (None, 1)                 0         
_________________________________________________________________
dense_3 (Dense)              (None, 1024)              2048      
_________________________________________________________________
dense_4 (Dense)              (None, 1024)              1049600   
_________________________________________________________________
dense_5 (Dense)              (None, 1024)              1049600   
_________________________________________________________________
dense_6 (Dense)              (None, 1)                 1025      
=================================================================
Total params: 2,102,273
Trainable params: 2,102,273
Non-trainable params: 0
_________________________________________________________________

Gradient Boost Regressor
{
  "alpha": 0.9,
  "criterion": "friedman_mse",
  "init": null,
  "learning_rate": 0.1,
  "loss": "ls",
  "max_depth": 5,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_impurity_split": null,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "n_estimators": 100,
  "n_iter_no_change": null,
  "presort": "auto",
  "random_state": null,
  "subsample": 1.0,
  "tol": 0.0001,
  "validation_fraction": 0.3,
  "verbose": 1,
  "warm_start": false
}
      Iter       Train Loss   Remaining Time 
         1           0.4074            0.00s
         2           0.3326            0.00s
         3           0.2716            0.03s
         4           0.2217            0.02s
         5           0.1811            0.02s
         6           0.1479            0.03s
         7           0.1209            0.03s
         8           0.0990            0.02s
         9           0.0809            0.03s
        10           0.0663            0.03s
        20           0.0091            0.03s
        30           0.0013            0.02s
        40           0.0002            0.02s
        50           0.0001            0.02s
        60           0.0000            0.01s
        70           0.0000            0.01s
        80           0.0000            0.01s
        90           0.0000            0.00s
       100           0.0000            0.00s

Polynomial Regressor
{
  "copy_X": true,
  "fit_intercept": true,
  "n_jobs": null,
  "normalize": false
}

Shallow Neural Network
Model: "model_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_3 (InputLayer)         (None, 1)                 0         
_________________________________________________________________
dense_7 (Dense)              (None, 1024)              2048      
_________________________________________________________________
dense_8 (Dense)              (None, 1)                 1025      
=================================================================
Total params: 3,073
Trainable params: 3,073
Non-trainable params: 0
_________________________________________________________________

Deep Neural Network
Model: "model_4"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_4 (InputLayer)         (None, 1)                 0         
_________________________________________________________________
dense_9 (Dense)              (None, 1024)              2048      
_________________________________________________________________
dense_10 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dense_11 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dense_12 (Dense)             (None, 1)                 1025      
=================================================================
Total params: 2,102,273
Trainable params: 2,102,273
Non-trainable params: 0
_________________________________________________________________

Gradient Boost Regressor
{
  "alpha": 0.9,
  "criterion": "friedman_mse",
  "init": null,
  "learning_rate": 0.1,
  "loss": "ls",
  "max_depth": 5,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_impurity_split": null,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "n_estimators": 100,
  "n_iter_no_change": null,
  "presort": "auto",
  "random_state": null,
  "subsample": 1.0,
  "tol": 0.0001,
  "validation_fraction": 0.3,
  "verbose": 1,
  "warm_start": false
}
      Iter       Train Loss   Remaining Time 
         1           0.4100            0.00s
         2           0.3350            0.00s
         3           0.2741            0.03s
         4           0.2244            0.02s
         5           0.1841            0.04s
         6           0.1510            0.03s
         7           0.1241            0.03s
         8           0.1020            0.03s
         9           0.0841            0.03s
        10           0.0694            0.03s
        20           0.0120            0.02s
        30           0.0040            0.02s
        40           0.0028            0.02s
        50           0.0024            0.02s
        60           0.0023            0.01s
        70           0.0022            0.01s
        80           0.0021            0.01s
        90           0.0020            0.00s
       100           0.0019            0.00s

Polynomial Regressor
{
  "copy_X": true,
  "fit_intercept": true,
  "n_jobs": null,
  "normalize": false
}

Shallow Neural Network
Model: "model_5"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_5 (InputLayer)         (None, 1)                 0         
_________________________________________________________________
dense_13 (Dense)             (None, 1024)              2048      
_________________________________________________________________
dense_14 (Dense)             (None, 1)                 1025      
=================================================================
Total params: 3,073
Trainable params: 3,073
Non-trainable params: 0
_________________________________________________________________

Deep Neural Network
Model: "model_6"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_6 (InputLayer)         (None, 1)                 0         
_________________________________________________________________
dense_15 (Dense)             (None, 1024)              2048      
_________________________________________________________________
dense_16 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dense_17 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dense_18 (Dense)             (None, 1)                 1025      
=================================================================
Total params: 2,102,273
Trainable params: 2,102,273
Non-trainable params: 0
_________________________________________________________________

Gradient Boost Regressor
{
  "alpha": 0.9,
  "criterion": "friedman_mse",
  "init": null,
  "learning_rate": 0.1,
  "loss": "ls",
  "max_depth": 5,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_impurity_split": null,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "n_estimators": 100,
  "n_iter_no_change": null,
  "presort": "auto",
  "random_state": null,
  "subsample": 1.0,
  "tol": 0.0001,
  "validation_fraction": 0.3,
  "verbose": 1,
  "warm_start": false
}
      Iter       Train Loss   Remaining Time 
         1           0.4236            0.10s
         2           0.3482            0.05s
         3           0.2867            0.06s
         4           0.2370            0.05s
         5           0.1963            0.04s
         6           0.1630            0.05s
         7           0.1360            0.04s
         8           0.1138            0.03s
         9           0.0955            0.04s
        10           0.0807            0.04s
        20           0.0224            0.03s
        30           0.0137            0.03s
        40           0.0118            0.02s
        50           0.0109            0.02s
        60           0.0104            0.01s
        70           0.0100            0.01s
        80           0.0096            0.01s
        90           0.0093            0.00s
       100           0.0089            0.00s

Polynomial Regressor
{
  "copy_X": true,
  "fit_intercept": true,
  "n_jobs": null,
  "normalize": false
}

Shallow Neural Network
Model: "model_7"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_7 (InputLayer)         (None, 1)                 0         
_________________________________________________________________
dense_19 (Dense)             (None, 1024)              2048      
_________________________________________________________________
dense_20 (Dense)             (None, 1)                 1025      
=================================================================
Total params: 3,073
Trainable params: 3,073
Non-trainable params: 0
_________________________________________________________________

Deep Neural Network
Model: "model_8"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_8 (InputLayer)         (None, 1)                 0         
_________________________________________________________________
dense_21 (Dense)             (None, 1024)              2048      
_________________________________________________________________
dense_22 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dense_23 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dense_24 (Dense)             (None, 1)                 1025      
=================================================================
Total params: 2,102,273
Trainable params: 2,102,273
Non-trainable params: 0
_________________________________________________________________

Gradient Boost Regressor
{
  "alpha": 0.9,
  "criterion": "friedman_mse",
  "init": null,
  "learning_rate": 0.1,
  "loss": "ls",
  "max_depth": 5,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_impurity_split": null,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "n_estimators": 100,
  "n_iter_no_change": null,
  "presort": "auto",
  "random_state": null,
  "subsample": 1.0,
  "tol": 0.0001,
  "validation_fraction": 0.3,
  "verbose": 1,
  "warm_start": false
}
      Iter       Train Loss   Remaining Time 
         1           0.4443            0.00s
         2           0.3692            0.05s
         3           0.3072            0.03s
         4           0.2569            0.02s
         5           0.2157            0.04s
         6           0.1823            0.03s
         7           0.1547            0.03s
         8           0.1321            0.03s
         9           0.1139            0.03s
        10           0.0989            0.03s
        20           0.0400            0.03s
        30           0.0301            0.02s
        40           0.0270            0.02s
        50           0.0255            0.02s
        60           0.0243            0.01s
        70           0.0235            0.01s
        80           0.0224            0.01s
        90           0.0214            0.00s
       100           0.0204            0.00s

Polynomial Regressor
{
  "copy_X": true,
  "fit_intercept": true,
  "n_jobs": null,
  "normalize": false
}

Shallow Neural Network
Model: "model_9"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_9 (InputLayer)         (None, 1)                 0         
_________________________________________________________________
dense_25 (Dense)             (None, 1024)              2048      
_________________________________________________________________
dense_26 (Dense)             (None, 1)                 1025      
=================================================================
Total params: 3,073
Trainable params: 3,073
Non-trainable params: 0
_________________________________________________________________

Deep Neural Network
Model: "model_10"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_10 (InputLayer)        (None, 1)                 0         
_________________________________________________________________
dense_27 (Dense)             (None, 1024)              2048      
_________________________________________________________________
dense_28 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dense_29 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dense_30 (Dense)             (None, 1)                 1025      
=================================================================
Total params: 2,102,273
Trainable params: 2,102,273
Non-trainable params: 0
_________________________________________________________________

Gradient Boost Regressor
{
  "alpha": 0.9,
  "criterion": "friedman_mse",
  "init": null,
  "learning_rate": 0.1,
  "loss": "ls",
  "max_depth": 5,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_impurity_split": null,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "n_estimators": 100,
  "n_iter_no_change": null,
  "presort": "auto",
  "random_state": null,
  "subsample": 1.0,
  "tol": 0.0001,
  "validation_fraction": 0.3,
  "verbose": 1,
  "warm_start": false
}
      Iter       Train Loss   Remaining Time 
         1           0.4768            0.10s
         2           0.4000            0.05s
         3           0.3377            0.06s
         4           0.2872            0.05s
         5           0.2456            0.04s
         6           0.2119            0.05s
         7           0.1842            0.04s
         8           0.1616            0.03s
         9           0.1431            0.04s
        10           0.1280            0.04s
        20           0.0666            0.03s
        30           0.0549            0.03s
        40           0.0505            0.02s
        50           0.0479            0.02s
        60           0.0456            0.01s
        70           0.0432            0.01s
        80           0.0408            0.01s
        90           0.0390            0.00s
       100           0.0375            0.00s

Polynomial Regressor
{
  "copy_X": true,
  "fit_intercept": true,
  "n_jobs": null,
  "normalize": false
}

Shallow Neural Network
Model: "model_11"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_11 (InputLayer)        (None, 1)                 0         
_________________________________________________________________
dense_31 (Dense)             (None, 1024)              2048      
_________________________________________________________________
dense_32 (Dense)             (None, 1)                 1025      
=================================================================
Total params: 3,073
Trainable params: 3,073
Non-trainable params: 0
_________________________________________________________________

Deep Neural Network
Model: "model_12"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_12 (InputLayer)        (None, 1)                 0         
_________________________________________________________________
dense_33 (Dense)             (None, 1024)              2048      
_________________________________________________________________
dense_34 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dense_35 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dense_36 (Dense)             (None, 1)                 1025      
=================================================================
Total params: 2,102,273
Trainable params: 2,102,273
Non-trainable params: 0
_________________________________________________________________

Gradient Boost Regressor
{
  "alpha": 0.9,
  "criterion": "friedman_mse",
  "init": null,
  "learning_rate": 0.1,
  "loss": "ls",
  "max_depth": 5,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_impurity_split": null,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "n_estimators": 100,
  "n_iter_no_change": null,
  "presort": "auto",
  "random_state": null,
  "subsample": 1.0,
  "tol": 0.0001,
  "validation_fraction": 0.3,
  "verbose": 1,
  "warm_start": false
}
      Iter       Train Loss   Remaining Time 
         1           0.5163            0.00s
         2           0.4380            0.00s
         3           0.3745            0.03s
         4           0.3227            0.02s
         5           0.2807            0.02s
         6           0.2459            0.03s
         7           0.2179            0.03s
         8           0.1947            0.02s
         9           0.1758            0.03s
        10           0.1603            0.03s
        20           0.0968            0.02s
        30           0.0834            0.02s
        40           0.0773            0.02s
        50           0.0740            0.02s
        60           0.0713            0.01s
        70           0.0682            0.01s
        80           0.0646            0.01s
        90           0.0621            0.00s
       100           0.0588            0.00s

Polynomial Regressor
{
  "copy_X": true,
  "fit_intercept": true,
  "n_jobs": null,
  "normalize": false
}

Shallow Neural Network
Model: "model_13"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_13 (InputLayer)        (None, 1)                 0         
_________________________________________________________________
dense_37 (Dense)             (None, 1024)              2048      
_________________________________________________________________
dense_38 (Dense)             (None, 1)                 1025      
=================================================================
Total params: 3,073
Trainable params: 3,073
Non-trainable params: 0
_________________________________________________________________

Deep Neural Network
Model: "model_14"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_14 (InputLayer)        (None, 1)                 0         
_________________________________________________________________
dense_39 (Dense)             (None, 1024)              2048      
_________________________________________________________________
dense_40 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dense_41 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dense_42 (Dense)             (None, 1)                 1025      
=================================================================
Total params: 2,102,273
Trainable params: 2,102,273
Non-trainable params: 0
_________________________________________________________________

Gradient Boost Regressor
{
  "alpha": 0.9,
  "criterion": "friedman_mse",
  "init": null,
  "learning_rate": 0.1,
  "loss": "ls",
  "max_depth": 5,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_impurity_split": null,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "n_estimators": 100,
  "n_iter_no_change": null,
  "presort": "auto",
  "random_state": null,
  "subsample": 1.0,
  "tol": 0.0001,
  "validation_fraction": 0.3,
  "verbose": 1,
  "warm_start": false
}
      Iter       Train Loss   Remaining Time 
         1           0.5766            0.00s
         2           0.4958            0.00s
         3           0.4294            0.03s
         4           0.3751            0.02s
         5           0.3307            0.02s
         6           0.2942            0.03s
         7           0.2648            0.03s
         8           0.2407            0.03s
         9           0.2207            0.03s
        10           0.2047            0.03s
        20           0.1356            0.03s
        30           0.1188            0.02s
        40           0.1102            0.02s
        50           0.1052            0.02s
        60           0.1005            0.01s
        70           0.0956            0.01s
        80           0.0911            0.01s
        90           0.0868            0.00s
       100           0.0828            0.00s

Polynomial Regressor
{
  "copy_X": true,
  "fit_intercept": true,
  "n_jobs": null,
  "normalize": false
}

Shallow Neural Network
Model: "model_15"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_15 (InputLayer)        (None, 1)                 0         
_________________________________________________________________
dense_43 (Dense)             (None, 1024)              2048      
_________________________________________________________________
dense_44 (Dense)             (None, 1)                 1025      
=================================================================
Total params: 3,073
Trainable params: 3,073
Non-trainable params: 0
_________________________________________________________________

Deep Neural Network
Model: "model_16"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_16 (InputLayer)        (None, 1)                 0         
_________________________________________________________________
dense_45 (Dense)             (None, 1024)              2048      
_________________________________________________________________
dense_46 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dense_47 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dense_48 (Dense)             (None, 1)                 1025      
=================================================================
Total params: 2,102,273
Trainable params: 2,102,273
Non-trainable params: 0
_________________________________________________________________

Gradient Boost Regressor
{
  "alpha": 0.9,
  "criterion": "friedman_mse",
  "init": null,
  "learning_rate": 0.1,
  "loss": "ls",
  "max_depth": 5,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_impurity_split": null,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "n_estimators": 100,
  "n_iter_no_change": null,
  "presort": "auto",
  "random_state": null,
  "subsample": 1.0,
  "tol": 0.0001,
  "validation_fraction": 0.3,
  "verbose": 1,
  "warm_start": false
}
      Iter       Train Loss   Remaining Time 
         1           0.6251            0.00s
         2           0.5451            0.00s
         3           0.4802            0.03s
         4           0.4272            0.02s
         5           0.3837            0.02s
         6           0.3488            0.03s
         7           0.3195            0.03s
         8           0.2954            0.02s
         9           0.2756            0.03s
        10           0.2596            0.03s
        20           0.1877            0.02s
        30           0.1670            0.02s
        40           0.1565            0.02s
        50           0.1495            0.02s
        60           0.1418            0.01s
        70           0.1362            0.01s
        80           0.1299            0.01s
        90           0.1241            0.00s
       100           0.1185            0.00s

Polynomial Regressor
{
  "copy_X": true,
  "fit_intercept": true,
  "n_jobs": null,
  "normalize": false
}

Shallow Neural Network
Model: "model_17"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_17 (InputLayer)        (None, 1)                 0         
_________________________________________________________________
dense_49 (Dense)             (None, 1024)              2048      
_________________________________________________________________
dense_50 (Dense)             (None, 1)                 1025      
=================================================================
Total params: 3,073
Trainable params: 3,073
Non-trainable params: 0
_________________________________________________________________

Deep Neural Network
Model: "model_18"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_18 (InputLayer)        (None, 1)                 0         
_________________________________________________________________
dense_51 (Dense)             (None, 1024)              2048      
_________________________________________________________________
dense_52 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dense_53 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dense_54 (Dense)             (None, 1)                 1025      
=================================================================
Total params: 2,102,273
Trainable params: 2,102,273
Non-trainable params: 0
_________________________________________________________________

Gradient Boost Regressor
{
  "alpha": 0.9,
  "criterion": "friedman_mse",
  "init": null,
  "learning_rate": 0.1,
  "loss": "ls",
  "max_depth": 5,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_impurity_split": null,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "n_estimators": 100,
  "n_iter_no_change": null,
  "presort": "auto",
  "random_state": null,
  "subsample": 1.0,
  "tol": 0.0001,
  "validation_fraction": 0.3,
  "verbose": 1,
  "warm_start": false
}
      Iter       Train Loss   Remaining Time 
         1           0.6868            0.00s
         2           0.6038            0.05s
         3           0.5364            0.03s
         4           0.4814            0.02s
         5           0.4362            0.04s
         6           0.3993            0.03s
         7           0.3694            0.03s
         8           0.3439            0.03s
         9           0.3234            0.03s
        10           0.3066            0.03s
        20           0.2321            0.03s
        30           0.2047            0.02s
        40           0.1913            0.02s
        50           0.1803            0.02s
        60           0.1694            0.01s
        70           0.1619            0.01s
        80           0.1544            0.01s
        90           0.1485            0.00s
       100           0.1414            0.00s

Polynomial Regressor
{
  "copy_X": true,
  "fit_intercept": true,
  "n_jobs": null,
  "normalize": false
}

Shallow Neural Network
Model: "model_19"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_19 (InputLayer)        (None, 1)                 0         
_________________________________________________________________
dense_55 (Dense)             (None, 1024)              2048      
_________________________________________________________________
dense_56 (Dense)             (None, 1)                 1025      
=================================================================
Total params: 3,073
Trainable params: 3,073
Non-trainable params: 0
_________________________________________________________________

Deep Neural Network
Model: "model_20"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_20 (InputLayer)        (None, 1)                 0         
_________________________________________________________________
dense_57 (Dense)             (None, 1024)              2048      
_________________________________________________________________
dense_58 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dense_59 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dense_60 (Dense)             (None, 1)                 1025      
=================================================================
Total params: 2,102,273
Trainable params: 2,102,273
Non-trainable params: 0
_________________________________________________________________

Gradient Boost Regressor
{
  "alpha": 0.9,
  "criterion": "friedman_mse",
  "init": null,
  "learning_rate": 0.1,
  "loss": "ls",
  "max_depth": 5,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_impurity_split": null,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "n_estimators": 100,
  "n_iter_no_change": null,
  "presort": "auto",
  "random_state": null,
  "subsample": 1.0,
  "tol": 0.0001,
  "validation_fraction": 0.3,
  "verbose": 1,
  "warm_start": false
}
      Iter       Train Loss   Remaining Time 
         1           0.7725            0.10s
         2           0.6847            0.05s
         3           0.6124            0.06s
         4           0.5533            0.05s
         5           0.5049            0.04s
         6           0.4643            0.05s
         7           0.4303            0.04s
         8           0.4026            0.03s
         9           0.3797            0.04s
        10           0.3607            0.04s
        20           0.2722            0.03s
        30           0.2412            0.03s
        40           0.2261            0.02s
        50           0.2125            0.02s
        60           0.2035            0.01s
        70           0.1949            0.01s
        80           0.1880            0.01s
        90           0.1797            0.00s
       100           0.1699            0.00s

Polynomial Regressor
{
  "copy_X": true,
  "fit_intercept": true,
  "n_jobs": null,
  "normalize": false
}

Shallow Neural Network
Model: "model_21"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_21 (InputLayer)        (None, 1)                 0         
_________________________________________________________________
dense_61 (Dense)             (None, 1024)              2048      
_________________________________________________________________
dense_62 (Dense)             (None, 1)                 1025      
=================================================================
Total params: 3,073
Trainable params: 3,073
Non-trainable params: 0
_________________________________________________________________

Deep Neural Network
Model: "model_22"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_22 (InputLayer)        (None, 1)                 0         
_________________________________________________________________
dense_63 (Dense)             (None, 1024)              2048      
_________________________________________________________________
dense_64 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dense_65 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dense_66 (Dense)             (None, 1)                 1025      
=================================================================
Total params: 2,102,273
Trainable params: 2,102,273
Non-trainable params: 0
_________________________________________________________________

Gradient Boost Regressor
{
  "alpha": 0.9,
  "criterion": "friedman_mse",
  "init": null,
  "learning_rate": 0.1,
  "loss": "ls",
  "max_depth": 5,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_impurity_split": null,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "n_estimators": 100,
  "n_iter_no_change": null,
  "presort": "auto",
  "random_state": null,
  "subsample": 1.0,
  "tol": 0.0001,
  "validation_fraction": 0.3,
  "verbose": 1,
  "warm_start": false
}
      Iter       Train Loss   Remaining Time 
         1           0.8555            0.00s
         2           0.7701            0.00s
         3           0.7005            0.03s
         4           0.6431            0.02s
         5           0.5966            0.04s
         6           0.5574            0.03s
         7           0.5251            0.03s
         8           0.4988            0.03s
         9           0.4774            0.03s
        10           0.4591            0.04s
        20           0.3709            0.03s
        30           0.3398            0.02s
        40           0.3203            0.02s
        50           0.3053            0.02s
        60           0.2895            0.01s
        70           0.2791            0.01s
        80           0.2667            0.01s
        90           0.2541            0.00s
       100           0.2434            0.00s

Polynomial Regressor
{
  "copy_X": true,
  "fit_intercept": true,
  "n_jobs": null,
  "normalize": false
}

Shallow Neural Network
Model: "model_23"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_23 (InputLayer)        (None, 1)                 0         
_________________________________________________________________
dense_67 (Dense)             (None, 1024)              2048      
_________________________________________________________________
dense_68 (Dense)             (None, 1)                 1025      
=================================================================
Total params: 3,073
Trainable params: 3,073
Non-trainable params: 0
_________________________________________________________________

Deep Neural Network
Model: "model_24"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_24 (InputLayer)        (None, 1)                 0         
_________________________________________________________________
dense_69 (Dense)             (None, 1024)              2048      
_________________________________________________________________
dense_70 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dense_71 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dense_72 (Dense)             (None, 1)                 1025      
=================================================================
Total params: 2,102,273
Trainable params: 2,102,273
Non-trainable params: 0
_________________________________________________________________

Gradient Boost Regressor
{
  "alpha": 0.9,
  "criterion": "friedman_mse",
  "init": null,
  "learning_rate": 0.1,
  "loss": "ls",
  "max_depth": 5,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_impurity_split": null,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "n_estimators": 100,
  "n_iter_no_change": null,
  "presort": "auto",
  "random_state": null,
  "subsample": 1.0,
  "tol": 0.0001,
  "validation_fraction": 0.3,
  "verbose": 1,
  "warm_start": false
}
      Iter       Train Loss   Remaining Time 
         1           0.9628            0.00s
         2           0.8705            0.05s
         3           0.7955            0.03s
         4           0.7341            0.05s
         5           0.6825            0.04s
         6           0.6399            0.03s
         7           0.6053            0.04s
         8           0.5772            0.03s
         9           0.5535            0.03s
        10           0.5341            0.04s
        20           0.4343            0.03s
        30           0.3969            0.02s
        40           0.3744            0.02s
        50           0.3565            0.02s
        60           0.3395            0.01s
        70           0.3236            0.01s
        80           0.3115            0.01s
        90           0.2954            0.00s
       100           0.2839            0.00s

Polynomial Regressor
{
  "copy_X": true,
  "fit_intercept": true,
  "n_jobs": null,
  "normalize": false
}

Shallow Neural Network
Model: "model_25"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_25 (InputLayer)        (None, 1)                 0         
_________________________________________________________________
dense_73 (Dense)             (None, 1024)              2048      
_________________________________________________________________
dense_74 (Dense)             (None, 1)                 1025      
=================================================================
Total params: 3,073
Trainable params: 3,073
Non-trainable params: 0
_________________________________________________________________

Deep Neural Network
Model: "model_26"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_26 (InputLayer)        (None, 1)                 0         
_________________________________________________________________
dense_75 (Dense)             (None, 1024)              2048      
_________________________________________________________________
dense_76 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dense_77 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dense_78 (Dense)             (None, 1)                 1025      
=================================================================
Total params: 2,102,273
Trainable params: 2,102,273
Non-trainable params: 0
_________________________________________________________________

Gradient Boost Regressor
{
  "alpha": 0.9,
  "criterion": "friedman_mse",
  "init": null,
  "learning_rate": 0.1,
  "loss": "ls",
  "max_depth": 5,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_impurity_split": null,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "n_estimators": 100,
  "n_iter_no_change": null,
  "presort": "auto",
  "random_state": null,
  "subsample": 1.0,
  "tol": 0.0001,
  "validation_fraction": 0.3,
  "verbose": 1,
  "warm_start": false
}
      Iter       Train Loss   Remaining Time 
         1           1.0898            0.10s
         2           0.9916            0.05s
         3           0.9114            0.06s
         4           0.8456            0.05s
         5           0.7924            0.04s
         6           0.7471            0.05s
         7           0.7109            0.04s
         8           0.6810            0.03s
         9           0.6535            0.04s
        10           0.6311            0.04s
        20           0.5292            0.03s
        30           0.4920            0.02s
        40           0.4655            0.02s
        50           0.4478            0.02s
        60           0.4265            0.01s
        70           0.4037            0.01s
        80           0.3906            0.01s
        90           0.3737            0.00s
       100           0.3582            0.00s

Polynomial Regressor
{
  "copy_X": true,
  "fit_intercept": true,
  "n_jobs": null,
  "normalize": false
}

Shallow Neural Network
Model: "model_27"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_27 (InputLayer)        (None, 1)                 0         
_________________________________________________________________
dense_79 (Dense)             (None, 1024)              2048      
_________________________________________________________________
dense_80 (Dense)             (None, 1)                 1025      
=================================================================
Total params: 3,073
Trainable params: 3,073
Non-trainable params: 0
_________________________________________________________________

Deep Neural Network
Model: "model_28"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_28 (InputLayer)        (None, 1)                 0         
_________________________________________________________________
dense_81 (Dense)             (None, 1024)              2048      
_________________________________________________________________
dense_82 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dense_83 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dense_84 (Dense)             (None, 1)                 1025      
=================================================================
Total params: 2,102,273
Trainable params: 2,102,273
Non-trainable params: 0
_________________________________________________________________

Gradient Boost Regressor
{
  "alpha": 0.9,
  "criterion": "friedman_mse",
  "init": null,
  "learning_rate": 0.1,
  "loss": "ls",
  "max_depth": 5,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_impurity_split": null,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "n_estimators": 100,
  "n_iter_no_change": null,
  "presort": "auto",
  "random_state": null,
  "subsample": 1.0,
  "tol": 0.0001,
  "validation_fraction": 0.3,
  "verbose": 1,
  "warm_start": false
}
      Iter       Train Loss   Remaining Time 
         1           1.0821            0.00s
         2           0.9940            0.00s
         3           0.9226            0.03s
         4           0.8639            0.02s
         5           0.8147            0.02s
         6           0.7730            0.03s
         7           0.7392            0.03s
         8           0.7119            0.03s
         9           0.6858            0.03s
        10           0.6655            0.03s
        20           0.5651            0.02s
        30           0.5242            0.02s
        40           0.4934            0.02s
        50           0.4667            0.01s
        60           0.4493            0.01s
        70           0.4264            0.01s
        80           0.4097            0.01s
        90           0.3927            0.00s
       100           0.3773            0.00s

Polynomial Regressor
{
  "copy_X": true,
  "fit_intercept": true,
  "n_jobs": null,
  "normalize": false
}

Shallow Neural Network
Model: "model_29"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_29 (InputLayer)        (None, 1)                 0         
_________________________________________________________________
dense_85 (Dense)             (None, 1024)              2048      
_________________________________________________________________
dense_86 (Dense)             (None, 1)                 1025      
=================================================================
Total params: 3,073
Trainable params: 3,073
Non-trainable params: 0
_________________________________________________________________

Deep Neural Network
Model: "model_30"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_30 (InputLayer)        (None, 1)                 0         
_________________________________________________________________
dense_87 (Dense)             (None, 1024)              2048      
_________________________________________________________________
dense_88 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dense_89 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dense_90 (Dense)             (None, 1)                 1025      
=================================================================
Total params: 2,102,273
Trainable params: 2,102,273
Non-trainable params: 0
_________________________________________________________________

Gradient Boost Regressor
{
  "alpha": 0.9,
  "criterion": "friedman_mse",
  "init": null,
  "learning_rate": 0.1,
  "loss": "ls",
  "max_depth": 5,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_impurity_split": null,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "n_estimators": 100,
  "n_iter_no_change": null,
  "presort": "auto",
  "random_state": null,
  "subsample": 1.0,
  "tol": 0.0001,
  "validation_fraction": 0.3,
  "verbose": 1,
  "warm_start": false
}
      Iter       Train Loss   Remaining Time 
         1           1.1559            0.10s
         2           1.0731            0.05s
         3           1.0053            0.03s
         4           0.9500            0.05s
         5           0.9041            0.04s
         6           0.8655            0.03s
         7           0.8321            0.04s
         8           0.8036            0.03s
         9           0.7786            0.03s
        10           0.7584            0.04s
        20           0.6479            0.03s
        30           0.5879            0.02s
        40           0.5579            0.02s
        50           0.5370            0.02s
        60           0.5145            0.01s
        70           0.4910            0.01s
        80           0.4652            0.01s
        90           0.4411            0.00s
       100           0.4211            0.00s

Polynomial Regressor
{
  "copy_X": true,
  "fit_intercept": true,
  "n_jobs": null,
  "normalize": false
}

Shallow Neural Network
Model: "model_31"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_31 (InputLayer)        (None, 1)                 0         
_________________________________________________________________
dense_91 (Dense)             (None, 1024)              2048      
_________________________________________________________________
dense_92 (Dense)             (None, 1)                 1025      
=================================================================
Total params: 3,073
Trainable params: 3,073
Non-trainable params: 0
_________________________________________________________________

Deep Neural Network
Model: "model_32"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_32 (InputLayer)        (None, 1)                 0         
_________________________________________________________________
dense_93 (Dense)             (None, 1024)              2048      
_________________________________________________________________
dense_94 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dense_95 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dense_96 (Dense)             (None, 1)                 1025      
=================================================================
Total params: 2,102,273
Trainable params: 2,102,273
Non-trainable params: 0
_________________________________________________________________

Gradient Boost Regressor
{
  "alpha": 0.9,
  "criterion": "friedman_mse",
  "init": null,
  "learning_rate": 0.1,
  "loss": "ls",
  "max_depth": 5,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_impurity_split": null,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "n_estimators": 100,
  "n_iter_no_change": null,
  "presort": "auto",
  "random_state": null,
  "subsample": 1.0,
  "tol": 0.0001,
  "validation_fraction": 0.3,
  "verbose": 1,
  "warm_start": false
}
      Iter       Train Loss   Remaining Time 
         1           1.3352            0.10s
         2           1.2444            0.05s
         3           1.1685            0.03s
         4           1.1058            0.05s
         5           1.0548            0.04s
         6           1.0134            0.03s
         7           0.9758            0.04s
         8           0.9442            0.03s
         9           0.9193            0.03s
        10           0.8988            0.04s
        20           0.7776            0.03s
        30           0.7223            0.02s
        40           0.6815            0.02s
        50           0.6564            0.02s
        60           0.6256            0.01s
        70           0.6021            0.01s
        80           0.5789            0.01s
        90           0.5512            0.00s
       100           0.5218            0.00s

Polynomial Regressor
{
  "copy_X": true,
  "fit_intercept": true,
  "n_jobs": null,
  "normalize": false
}
