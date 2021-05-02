import numpy as np 
from ctypes import *


# TODO : read path from a .env file or smtg 
dll_path = "C:/Users/ttres/Desktop/3A/S2/MachineLearning/ml_library/cmake-build-debug/ml_library.dll"
mylib = cdll.LoadLibrary(dll_path)


def create_model(model_dim):

    mylib.create_linear_model.argtypes = [c_int]
    mylib.create_linear_model.restype = POINTER(c_float)

    p_model = mylib.create_linear_model(model_dim)
    
    pretty = np.ctypeslib.as_array(p_model , (model_dim,))

    print(pretty)

    return p_model


def destroy_model(p_model):
    mylib.destroy_linear_model.argtypes = [POINTER(c_float)]
    mylib.destroy_linear_model.restype = None


def as_C_array(dataset):
    arr_size = len(dataset)

    arr_type = c_float * arr_size
    arr = arr_type(*dataset)

    return arr,arr_type


def train_model(p_model , model_dim , inputs , outputs , alpha = 0.001 , epochs = 2500):

    input_dataset,input_type = as_C_array(inputs)
    output_dataset,output_type = as_C_array(outputs)

    mylib.train_classification_rosenblatt_rule_linear_model.argtypes = [POINTER(c_float) , c_int , input_type , c_int , output_type , c_float , c_int ]
    mylib.train_classification_rosenblatt_rule_linear_model.restype = None

    mylib.train_classification_rosenblatt_rule_linear_model(p_model , model_dim , input_dataset , len(inputs) , output_dataset , alpha , epochs)

def predict_model_classif(p_model , model_dim , dataset):
    predict_dataset,predict_type = as_C_array(dataset)

    mylib.predict_linear_model_classification.argtypes = [POINTER(c_float) , c_int , predict_type]
    mylib.predict_linear_model_classification.restype = c_float

    predict_value = mylib.predict_linear_model_classification(p_model , model_dim , predict_dataset)

    return predict_value


inputs = [
    3, 4,
    6, 5,
    4, 7
]
outputs = [
    1,
    1,
    -1
]

model_dim = 2

p_model = create_model(model_dim)

test_before = predict_model_classif(p_model , model_dim , [7,7])

train_model(p_model , model_dim , inputs , outputs)

test_after = predict_model_classif(p_model , model_dim , [3,7])

destroy_model(p_model)

print("before : ", test_before , "And after : " , test_after)
