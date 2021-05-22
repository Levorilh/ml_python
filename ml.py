from linear import *
from mlp import *

init_random()
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


# TEST MLP

inputs = [
    [0, 0],
    [0, 1],
    [1, 1],
    [1, 0]
]
outputs = [
    -1,
    1,
    -1,
    1
]

npl = [2, 3, 1]
p_model = create_mlp_model(npl)
test_before = predict_mlp_model_classification(p_model, [0, 0])
print("test before:00", test_before)
destroy_mlp_prediction(test_before)
test_before = predict_mlp_model_classification(p_model, [0, 1])
print("test before:01", test_before)
destroy_mlp_prediction(test_before)
test_before = predict_mlp_model_classification(p_model, [1, 1])
print("test before:11", test_before)
destroy_mlp_prediction(test_before)
test_before = predict_mlp_model_classification(p_model, [1, 0])
print("test before:10", test_before)
destroy_mlp_prediction(test_before)

train_classification_stochastic_gradient_backpropagation_mlp_model(p_model,
                                                                   inputs,
                                                                   outputs
                                                                   )
test_after = predict_mlp_model_classification(p_model, [0, 0])
print("test after:00", test_after)
destroy_mlp_prediction(test_after)
test_after = predict_mlp_model_classification(p_model, [0, 1])
print("test after:01", test_after)
destroy_mlp_prediction(test_after)
test_after = predict_mlp_model_classification(p_model, [1, 1])
print("test after:11", test_after)
destroy_mlp_prediction(test_after)
test_after = predict_mlp_model_classification(p_model, [1, 0])
print("test after:10", test_after)
destroy_mlp_prediction(test_after)

test_after = predict_mlp_model_classification(p_model, [-1, 0])
print("test after:-10", test_after)
destroy_mlp_prediction(test_after)
test_after = predict_mlp_model_classification(p_model, [2, 0])
print("test after:21", test_after)
destroy_mlp_prediction(test_after)

destroy_mlp_model(p_model)
"""
