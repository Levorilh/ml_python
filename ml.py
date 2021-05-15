from linear import *
from mlp import *

"""
# TEST LINEAR

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

p_model = create_linear_model(model_dim)

test_before = predict_linear_model_classif(p_model , model_dim , [7,7])

train_linear_model(p_model , model_dim , inputs , outputs)

test_after = predict_linear_model_classif(p_model , model_dim , [7,7])

destroy_linear_model(p_model)

print("before : ", test_before , "And after : " , test_after)
"""

# TEST MLP

inputs = [
    [0, 0],
    [0, 1],
    [1, 1],
    [1, 0]
]
outputs = [
    -1,
    -1,
    1,
    1
]



npl = [2, 3, 1]
p_model = create_mlp_model(npl)
test_before = predict_mlp_model_classification(p_model, [0, 0])
print("test before:", test_before)

train_classification_stochastic_gradient_backpropagation_mlp_model(p_model,
                                                                   inputs,
                                                                   outputs
                                                                   )
test_after = predict_mlp_model_classification(p_model, [0, 0])
print("test after:", test_after)

destroy_mlp_model(p_model)
