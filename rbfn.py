import numpy as np
from ctypes import *
from settings import *

def as_C_array(dataset, datatype=c_float):
    arr_size = len(dataset)
    arr_type = datatype * arr_size
    arr = arr_type(*dataset)
    return arr, arr_type


def create_rbfn_model(input_dim, num_classes, k):
    mylib.create_rbfn_model.argtypes = [c_int, c_int, c_int]
    mylib.create_rbfn_model.restype = c_void_p
    return mylib.create_rbfn_model(input_dim, num_classes, k)


def destroy_rbfn_model(model):
    mylib.destroy_rbfn_model.argtypes = [c_void_p]
    mylib.destroy_rbfn_model.restype = None
    mylib.destroy_rbfn_model(model)


def train_rbfn_model(model, dataset_inputs, dataset_expected_outputs, naif=True, max_iters=100):
    samples_count = len(dataset_inputs)
    flattened_dataset_inputs = np.array(dataset_inputs).flatten()
    flattened_dataset_expected_outputs = np.array(dataset_expected_outputs).flatten()
    arr_input, input_type = as_C_array(flattened_dataset_inputs, c_double)
    arr_output, output_type = as_C_array(flattened_dataset_expected_outputs, c_double)
    mylib.train_rbfn_model.argtypes = [c_void_p, input_type, c_int, output_type, c_bool, c_int]
    mylib.train_rbfn_model.restype = None
    mylib.train_rbfn_model(model, arr_input, samples_count, arr_output, naif, max_iters)


def predict_rbfn(model, dataset_inputs):
    flattened_dataset_inputs = np.array(dataset_inputs).flatten()
    arr, arr_type = as_C_array(flattened_dataset_inputs, c_double)
    mylib.predict_rbfn.argtypes = [c_void_p, arr_type]
    mylib.predict_rbfn.restype = POINTER(c_double)
    predict_value = mylib.predict_rbfn(model, arr)
    rslt = list(np.ctypeslib.as_array(predict_value, (2,)))
    destroy_rbfn_prediction(predict_value)
    return rslt


def destroy_rbfn_prediction(prediction):
    mylib.destroy_rbfn_prediction.argtypes = [POINTER(c_double)]
    mylib.destroy_rbfn_prediction.restype = None
    mylib.destroy_rbfn_prediction(prediction)
