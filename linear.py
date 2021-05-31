import numpy as np
from ctypes import *
from settings import *

def create_linear_model(model_dim):
    mylib.create_linear_model.argtypes = [c_int]
    mylib.create_linear_model.restype = POINTER(c_float)

    p_model = mylib.create_linear_model(model_dim)

    pretty = np.ctypeslib.as_array(p_model, (model_dim+1,))
    print(pretty)

    return p_model


def destroy_linear_model(p_model):
    mylib.destroy_linear_model.argtypes = [POINTER(c_float)]
    mylib.destroy_linear_model.restype = None

    mylib.destroy_linear_model(p_model)


def as_C_array(dataset, datatype=c_float):
    arr_size = len(dataset)

    arr_type = datatype * arr_size
    arr = arr_type(*dataset)

    return arr, arr_type


def train_linear_model(p_model, model_dim, inputs, outputs, alpha=0.001, epochs=1000):
    if type(inputs) is list:
        inputs = np.array(inputs)
    inputs_flattened = inputs.flatten()

    input_dataset, input_type = as_C_array(inputs_flattened)
    output_dataset, output_type = as_C_array(outputs)

    mylib.train_classification_rosenblatt_rule_linear_model.argtypes = [POINTER(c_float), c_int, input_type, c_int,
                                                                        output_type, c_float, c_int]
    mylib.train_classification_rosenblatt_rule_linear_model.restype = None

    mylib.train_classification_rosenblatt_rule_linear_model(p_model, model_dim, input_dataset, len(inputs),
                                                            output_dataset, alpha, epochs)


def predict_linear_model_classif(p_model, model_dim, dataset):
    predict_dataset, predict_type = as_C_array(dataset)

    mylib.predict_linear_model_classification.argtypes = [POINTER(c_float), c_int, predict_type]
    mylib.predict_linear_model_classification.restype = c_float

    predict_value = mylib.predict_linear_model_classification(p_model, model_dim, predict_dataset)

    return predict_value


def predict_linear_model_regression(p_model, model_dim, dataset):
    predict_dataset, predict_type = as_C_array(dataset)

    mylib.predict_linear_model_regression.argtypes = [POINTER(c_float), c_int, predict_type]
    mylib.predict_linear_model_regression.restype = c_float

    predict_value = mylib.predict_linear_model_regression(p_model, model_dim, predict_dataset)

    return predict_value