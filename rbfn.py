import numpy as np
from ctypes import *
from settings import *

class Centroid(Structure):
    _fields_ = [
        ("label",c_int),
        ("coords",POINTER(c_double)),
        ("coord_count",c_int),
        ("deviation",c_double)
    ]

class RBF(Structure):
    _fields_ = [
        ("input_dim", c_int),
        ("num_classes", c_int),
        ("k", c_int),
        ("samples_count", c_int),
        ("W", POINTER(POINTER(c_float))),
        ("clusters", POINTER(POINTER(Centroid)))
    ]

def as_C_array(dataset, datatype=c_float):
    arr_size = len(dataset)
    arr_type = datatype * arr_size
    arr = arr_type(*dataset)
    return arr, arr_type

def create_rbfn_model(input_dim, num_classes, k) :
    mylib.create_rbfn_model.argtypes = [c_int, c_int, c_int]
    mylib.create_rbfn_model.restype = c_void_p
    return mylib.create_rbfn_model(input_dim, num_classes, k)

def destroy_rbfn_model(model) :
    mylib.destroy_rbfn_model.argtypes = [c_void_p]
    mylib.destroy_rbfn_model.restype = None
    mylib.destroy_rbfn_model(model)

def train_rbfn_model(model, dataset_inputs, dataset_expected_outputs, max_iters = 100) :
    samples_count = len(dataset_inputs)
    flattened_dataset_inputs = np.array(dataset_inputs).flatten()
    flattened_dataset_expected_outputs = np.array(dataset_expected_outputs).flatten()
    arr_input, input_type = as_C_array(flattened_dataset_inputs)
    arr_output, output_type = as_C_array(flattened_dataset_expected_outputs)
    mylib.train_rbfn_model.argtypes = [c_void_p, input_type, c_int, output_type, c_int]
    mylib.train_rbfn_model.restype = None
    mylib.train_rbfn_model(model, arr_input, samples_count, arr_output, max_iters)

def predict_rbfn(model, dataset_inputs) :
    flattened_dataset_inputs = np.array(dataset_inputs).flatten()
    arr, arr_type = as_C_array(flattened_dataset_inputs)
    mylib.predict_rbfn.argtypes = [c_void_p, arr_type]
    mylib.predict_rbfn.restype = POINTER(c_float)
    return mylib.predict_rbfn(model, arr)

def destroy_rbfn_prediction(prediction) :
    arr, arr_type = as_C_array(prediction)
    mylib.destroy_rbfn_prediction.argtypes = [arr_type]
    mylib.destroy_rbfn_prediction.restype = None
    mylib.destroy_rbfn_prediction(arr)