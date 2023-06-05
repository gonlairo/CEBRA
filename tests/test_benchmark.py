#
# (c) All rights reserved. ECOLE POLYTECHNIQUE FÉDÉRALE DE LAUSANNE,
# Switzerland, Laboratory of Prof. Mackenzie W. Mathis (UPMWMATHIS) and
# original authors: Steffen Schneider, Jin H Lee, Mackenzie W Mathis. 2023.
#
# Source code:
# https://github.com/AdaptiveMotorControlLab/CEBRA
#
# Please see LICENSE.md for the full license document:
# https://github.com/AdaptiveMotorControlLab/CEBRA/LICENSE.md
#
import numpy as np
import pytest
import sklearn
import sklearn.metrics
import sklearn.neighbors
import torch

import cebra.data
import cebra.datasets
import cebra.models
import cebra.solver
import joblib
import utils_benchmark

def _split_data(data, test_ratio=0.2):
    split_idx = int(len(data) * (1 - test_ratio))
    neural_train = data.neural[:split_idx]
    neural_test = data.neural[split_idx:]
    label_train = data.continuous_index[:split_idx]
    label_test = data.continuous_index[split_idx:]

    return (
        neural_train.numpy(),
        neural_test.numpy(),
        label_train.numpy(),
        label_test.numpy(),
    )


def _decode(emb_train, emb_test, label_train, label_test, n_neighbors=36):
    pos_decoder = sklearn.neighbors.KNeighborsRegressor(n_neighbors,
                                                        metric="cosine")
    dir_decoder = sklearn.neighbors.KNeighborsClassifier(n_neighbors,
                                                         metric="cosine")

    pos_decoder.fit(emb_train, label_train[:, 0])
    dir_decoder.fit(emb_train, label_train[:, 1])

    pos_pred = pos_decoder.predict(emb_test)
    dir_pred = dir_decoder.predict(emb_test)

    prediction = np.stack([pos_pred, dir_pred], axis=1)
    r2_score = sklearn.metrics.r2_score(label_test[:, :2], prediction)
    pos_err = np.median(abs(pos_pred - label_test[:, 0]))

    return r2_score, pos_err


def _eval(train_set, test_set, solver):
    emb_train = solver.transform(train_set[torch.arange(
        len(train_set))]).numpy()
    emb_test = solver.transform(test_set[torch.arange(len(test_set))]).numpy()
    label_train = train_set.continuous_index.numpy()
    label_test = test_set.continuous_index.numpy()
    r2_score, pos_err = _decode(emb_train, emb_test, label_train, label_test)

    return r2_score, pos_err


def _train(train_set, loader_initfunc, solver_initfunc, model_params, loader_params, lr):
    model = cebra.models.init(
        name = model_params["model_name"],
        num_neurons = train_set.neural.shape[-1],
        num_units = model_params["hidden_size"],
        num_output = model_params["output_size"],
    )
    train_loader = loader_initfunc(train_set, **loader_params)
    criterion = cebra.models.InfoNCE()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    solver = solver_initfunc(model=model,
                             criterion=criterion,
                             optimizer=optimizer)
    solver.fit(train_loader)

    return solver


def _run_hippocampus(data_name_dict, dataset_initfunc, loader_initfunc, solver_initfunc, lower_bound_score):
    results = {}
    embeddings = {}

    for mouse in data_name_dict.keys():
        results_mouse = {}
        dataset = cebra.datasets.init(f'rat-hippocampus-single-{mouse}')
        neural_train, neural_test, label_train, label_test = _split_data(
            dataset, test_ratio)
        offset = cebra.data.datatypes.Offset(loader_params["time_offset"] // 2,
                                             loader_params["time_offset"] // 2)
        train_set = dataset_initfunc(neural_train,
                                     continuous=label_train,
                                     offset=offset)
        valid_set = dataset_initfunc(neural_test,
                                     continuous=label_test,
                                     offset=offset)

        # TRAIN MODEL
        solver = _train(train_set, loader_initfunc, solver_initfunc)

        # decoding acc
        r2_score, pos_err = _eval(train_set, valid_set, solver)
        results_mouse.update({"r2" : r2_score, "pos_err" : pos_err})

        # goodness of fit (loss)
        goodness_of_fit = solver.history
        results_mouse.update({"goodness_of_fit" : goodness_of_fit})

        X = np.pad(train_set.neural, ((loader_params["time_offset"]//2, loader_params["time_offset"]//2 - 1), (0, 0)), mode="edge")
        X = torch.from_numpy(X).float()

        if isinstance(solver.model, cebra.models.ConvolutionalModelMixin):
            # Fully convolutional evaluation, switch (T, C) -> (1, C, T)
            X = X.transpose(1, 0).unsqueeze(0)
            embedding = solver.model(X).detach().cpu().numpy().squeeze(0).transpose(1, 0)
        else:
            # Standard evaluation, (T, C, dt)
            embedding = solver.model(X).detach().cpu().numpy()

        results_mouse.update({"embedding":embedding})
        results_mouse.update({"train_set":train_set})
        results[mouse] = results_mouse


    # consistency across datasets
    labels = [results[mouse]["train_set"].continuous_index[:, 0]
          for mouse in list(data_name_dict.keys())]

    embeddings = [results[mouse]["embedding"] for mouse in  list(data_name_dict.keys())]
    scores, pairs, subjects = cebra.sklearn.metrics.consistency_score(embeddings = embeddings,
                                                                      labels=labels,
                                                                      dataset_ids=list(data_name_dict.keys()),
                                                                      between="datasets")
    joblib.dump({"results": results, 
                 "consistency":(scores, pairs, subjects)}, 'results_benchmark.jl')
    
    ## TESTS: how exactly should I test for stuff?
    #assert r2_score > lower_bound_score["r2_score"] 
    #assert pos_err < lower_bound_score["median_error"]
    #assert goodness_of_fit < lower_bound_score["infonce"]


@torch.no_grad()
def get_emissions(model, dataset):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model.to(device)
    dataset.configure_for(model)
    return model(dataset[torch.arange(len(dataset))].to(device)).cpu().numpy()

def _compute_emissions_single(solver, dataset):
    return get_emissions(solver.model, dataset)


def _run_allen(cortex, num_neurons, seed, loader_initfunc, solver_initfunc, model_params, loader_params, lr):

    #ca_train = cebra.datasets.init(f'allen-movie-one-ca-{cortex}-{num_neurons}-train-10-{seed}')
    np_train = cebra.datasets.init(f'allen-movie-one-neuropixel-{cortex}-{num_neurons}-train-10-{seed}')

    #ca_test = cebra.datasets.init(f'allen-movie-one-ca-{cortex}-{num_neurons}-test-10-{seed}')
    np_test = cebra.datasets.init(f'allen-movie-one-neuropixel-{cortex}-{num_neurons}-test-10-{seed}')

    # TRAIN MODEL
    solver = _train(np_train, loader_initfunc, solver_initfunc, model_params, loader_params, lr)

    goodness_of_fit = solver.history

    # compute embedding for train and test
    cebra_np_train = _compute_emissions_single(solver, np_train)
    cebra_np_test =  _compute_emissions_single(solver, np_test)

    pred_cebra, errs_cebra ,acc_cebra = utils_benchmark.allen_frame_id_decode(cebra_np_train, np.tile(np.arange(900), 9),
                                                                              cebra_np_test, np.arange(900),
                                                                              modality = 'neuropixel', decoder = 'knn')

    print(f'CEBRA Neuropixel: {acc_cebra:.2f}%')
    print("GOF:", goodness_of_fit[-1])


###### RUN TEST
@pytest.mark.benchmark
def test_single_session_hippocampus(benchmark):

    model_params = {
    "n_neurons": 120,
    "hidden_size": 64,
    "output_size": 32,
    "model_name": "offset10-model",
    }
    loader_params = {
                    "conditional": "time_delta",
                    "num_steps": 10,
                    "batch_size": 512,
                    "time_offset": 10,
    }
    lr = 3e-4
    test_ratio = 0.2

    hippocampus_pos = {}
    hippocampus_pos["achilles"] = cebra.datasets.init('rat-hippocampus-single-achilles')
    hippocampus_pos["buddy"]    = cebra.datasets.init('rat-hippocampus-single-buddy')
    hippocampus_pos["cicero"]   = cebra.datasets.init('rat-hippocampus-single-cicero')
    hippocampus_pos["gatsby"]   = cebra.datasets.init('rat-hippocampus-single-gatsby')

    lower_bound_score = {"r2_score": 30, "median_error": 1.5, "infonce": 6.5}

    single_session_setup_hippocampus = {
                                        "data_name_dict": hippocampus_pos,
                                        "dataset_initfunc": cebra.data.TensorDataset,
                                        "loader_initfunc": cebra.data.ContinuousDataLoader,
                                        "solver_initfunc": cebra.solver.SingleSessionSolver,
                                        "lower_bound_score" : lower_bound_score,
                                        }
    
    # check if pedantic mode is exactly what we want
    benchmark.pedantic(_run_hippocampus, kwargs = single_session_setup_hippocampus,
                                         rounds=1)
    
## ALLEN DATASET!


@pytest.mark.benchmark
def test_single_session_allen(benchmark):

    model_params = {
                    "n_neurons": 120,
                    "hidden_size": 64,
                    "output_size": 32,
                    "model_name": "offset1-model", #TODO: only works with offset1 model
                    }
    
    loader_params = {
                    "conditional": "time_delta",
                    "num_steps": 10,
                    "batch_size": 512,
                    "time_offset": 10,
                }
    lr = 3e-4

    single_session_setup_allen = {
                                "cortex":'VISp',
                                "seed": 333,
                                "num_neurons": 800,
                                "loader_initfunc": cebra.data.ContinuousDataLoader,
                                "solver_initfunc": cebra.solver.SingleSessionSolver, 
                                "model_params": model_params,
                                "loader_params": loader_params,
                                "lr": lr   
    }

    benchmark.pedantic(_run_allen, 
                       kwargs = single_session_setup_allen,
                       rounds = 1)
    

