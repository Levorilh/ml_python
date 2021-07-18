import numpy as np
from ctypes import *
from settings import *


class MLP(Structure):
    _fields_ = [
        ("d_length", c_int),
        ("d", POINTER(c_float)),
        ("X", POINTER(POINTER(c_float))),
        ("deltas", POINTER(POINTER(c_float))),
        ("W", POINTER(POINTER(POINTER(c_float))))
    ]


def create_mlp_model(npl):
    # npl, npl_length = as_C_array(npl)
    npl_length = len(npl)
    npl = (c_int * npl_length)(*npl)

    mylib.create_mlp_model.argtypes = [POINTER(c_int), c_int]
    mylib.create_mlp_model.restype = c_void_p

    p_model = mylib.create_mlp_model(npl, npl_length)

    return p_model, npl[-1]


def destroy_mlp_model(p_model):
    mylib.destroy_mlp_model.argtypes = [c_void_p]
    mylib.destroy_mlp_model.restype = None

    mylib.destroy_mlp_model(p_model)


def destroy_mlp_prediction(prediction):
    mylib.destroy_mlp_prediction.argtypes = [POINTER(c_float)]
    mylib.destroy_mlp_prediction.restype = None

    mylib.destroy_mlp_prediction(prediction)


def as_C_array(dataset):
    arr_size = len(dataset)

    arr_type = c_float * arr_size
    arr = arr_type(*dataset)

    return arr, arr_type


def train_classification_stochastic_gradient_backpropagation_mlp_model(p_model,
                                                                       inputs,
                                                                       outputs,
                                                                       alpha=0.001,
                                                                       epochs=10000):
    samples_count = len(inputs)

    X = np.array(inputs)
    flattened_inputs = X.flatten()

    input_dataset, input_type = as_C_array(flattened_inputs)
    output_dataset, output_type = as_C_array(outputs)

    mylib.train_classification_stochastic_gradient_backpropagation_mlp_model.argtypes = [c_void_p,
                                                                                         input_type,
                                                                                         c_int,
                                                                                         output_type,
                                                                                         c_float,
                                                                                         c_int]
    mylib.train_classification_stochastic_gradient_backpropagation_mlp_model.restype = None

    mylib.train_classification_stochastic_gradient_backpropagation_mlp_model(p_model,
                                                                             input_dataset,
                                                                             samples_count,
                                                                             output_dataset,
                                                                             alpha,
                                                                             epochs)


def train_regression_stochastic_gradient_backpropagation_mlp_model(p_model,
                                                                   inputs,
                                                                   outputs,
                                                                   alpha=0.001,
                                                                   epochs=10000):
    X = np.array(inputs)
    flattened_inputs = X.flatten()

    input_dataset, input_type = as_C_array(flattened_inputs)
    output_dataset, output_type = as_C_array(outputs)

    mylib.train_regression_stochastic_gradient_backpropagation_mlp_model.argtypes = [c_void_p,
                                                                                     input_type,
                                                                                     c_int,
                                                                                     output_type,
                                                                                     c_float,
                                                                                     c_int]
    mylib.train_regression_stochastic_gradient_backpropagation_mlp_model.restype = None

    mylib.train_regression_stochastic_gradient_backpropagation_mlp_model(p_model,
                                                                         input_dataset,
                                                                         len(inputs),
                                                                         output_dataset,
                                                                         alpha,
                                                                         epochs)


def predict_mlp_model_classification(p_model, sample_input, last_layer_len=1):
    sample_input_ctype, si_type = as_C_array(sample_input)

    # si_length = len(sample_input)

    mylib.predict_mlp_model_classification.argtypes = [c_void_p, POINTER(si_type)]
    mylib.predict_mlp_model_classification.restype = POINTER(c_float)

    predict_value = mylib.predict_mlp_model_classification(p_model, sample_input_ctype)

    res = list(np.ctypeslib.as_array(predict_value, (last_layer_len,)))
    destroy_mlp_prediction(predict_value)

    return res


def predict_mlp_model_regression(p_model, sample_input):
    sample_input, si_type = as_C_array(sample_input)

    mylib.predict_mlp_model_regression.argtypes = [c_void_p, POINTER(si_type)]
    mylib.predict_mlp_model_regression.restype = POINTER(c_float)

    predict_value = mylib.predict_mlp_model_regression(p_model, sample_input)

    return predict_value


def init_random():
    mylib.init_random.argtypes = []
    mylib.init_random.restype = None

    mylib.init_random()


def load_mlp_model(filename):
    mylib.load_mlp_model.argtypes = [c_char_p]
    mylib.load_mlp_model.restype = c_void_p

    p_model = mylib.load_mlp_model(bytes(filename, 'utf-8'))

    if not p_model:
        print("le modele n'a pas pu être créé")
        raise ValueError()

    return p_model


def save_mlp_model(p_model, filename):
    mylib.save_mlp_model.argtypes = [c_void_p, c_char_p]
    mylib.save_mlp_model.restype = None

    mylib.save_mlp_model(p_model, bytes(filename, 'utf-8'))
