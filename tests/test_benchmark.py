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

model_params = {
    "n_neurons": 120,
    "hidden_size": 64,
    "output_size": 32,
    "model_name": "offset10-model",
}
loader_params = {
    "conditional": "time_delta",
    "num_steps": 5000,
    "batch_size": 512,
    "time_offset": 10,
}
lr = 3e-4
test_ratio = 0.2

single_session_setup = {
    "dataset_initfunc": cebra.data.TensorDataset,
    "loader_initfunc": cebra.data.ContinuousDataLoader,
    "solver_initfunc": cebra.solver.SingleSessionSolver,
}

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


def _train(train_set, loader_initfunc, solver_initfunc):
    model = cebra.models.init(
        model_params["model_name"],
        model_params["n_neurons"],
        model_params["hidden_size"],
        model_params["output_size"],
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

        # save the embeddings
        results_mouse.update({"embedding":solver.model(train_set)})

        embeddings[mouse] = solver.model(train_set)
        results[mouse] = results_mouse

        assert r2_score > lower_bound_score["r2_score"] 
        assert pos_err < lower_bound_score["median_error"]
        assert goodness_of_fit < lower_bound_score["infonce"]
    

    # TODO: compute consistency across animals!
    labels = [data_name_dict[mouse].continuous_index[:, 0]
          for mouse in list(data_name_dict.keys())]

    time_scores, time_pairs, time_subjects = cebra.sklearn.metrics.consistency_score(embeddings=list(embeddings.values()),
                                                                                 labels=labels,
                                                                                 dataset_ids=list(
                                                                                     embeddings.keys()),
                                                                                 between="datasets")

    print(f"{solver_initfunc.__name__}: r2 score = {r2_score:.4f}, "
          f"median abs error = {pos_err:.4f}")
    return None
    # What exactly should I return?
    #r2_score, pos_err


@pytest.mark.benchmark
def test_single_session_hippocampus(benchmark):

    ### TEST HIPPOCAMPUS
    hippocampus_pos = {}
    hippocampus_pos["achilles"] = cebra.datasets.init('rat-hippocampus-single-achilles')
    hippocampus_pos["buddy"] = cebra.datasets.init('rat-hippocampus-single-buddy')
    hippocampus_pos["cicero"] = cebra.datasets.init('rat-hippocampus-single-cicero')
    hippocampus_pos["gatsby"] = cebra.datasets.init('rat-hippocampus-single-gatsby')

    lower_bound_score = {"r2_score": 30, "median_error": 1.5, "infonce": 6.5}

    # check if pedantic mode is exactly what we want

    single_session_setup_hippocampus = {
    "data_name_dict":hippocampus_pos,
    "dataset_initfunc": cebra.data.TensorDataset,
    "loader_initfunc": cebra.data.ContinuousDataLoader,
    "solver_initfunc": cebra.solver.SingleSessionSolver,
    "lower_bound_score" : lower_bound_score,
}
    benchmark.pedantic(_run_hippocampus, kwargs = single_session_setup_hippocampus,
                                         rounds=1)
    

@pytest.mark.benchmark
def test_single_session_allen(benchmark):
    pass