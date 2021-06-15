from linear import *
from mlp import *

init_random()

"""
import numpy as np

# TEST LINEAR
test_before = []


inputs = np.array([
    [1, 1],
    [2, 3],
    [3, 3]
])

outputs = np.array([
    1,
    -1,
    -1
])
"""
"""
inputs = [
                [3, 4],
                [6, 5],
                [4, 7],
]

outputs = [
                1,
                1,
                -1,
]

model_dim = 2

errors = 0

for _ in range(50):
    test_after = []
    p_model = create_linear_model(model_dim)

    train_linear_model(p_model, model_dim, inputs, outputs, alpha=0.001, epochs=10_000)

    for data, expected in zip(inputs, outputs):
        out = predict_linear_model_classif(p_model, model_dim, data)
        test_after.append(out)
        if out != expected:
            errors += 1
    print(test_after)
    destroy_linear_model(p_model)

print(f"errors: {errors}")
"""
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
    1,
    -1,
    1
]

npl = [2, 3, 1]
p_model = create_mlp_model(npl)
test_before = predict_mlp_model_classification(p_model, [0, 0])
print("test before:00", test_before)
#destroy_mlp_prediction(test_before)
test_before = predict_mlp_model_classification(p_model, [0, 1])
print("test before:01", test_before)
#destroy_mlp_prediction(test_before)
test_before = predict_mlp_model_classification(p_model, [1, 1])
print("test before:11", test_before)
#destroy_mlp_prediction(test_before)
test_before = predict_mlp_model_classification(p_model, [1, 0])
print("test before:10", test_before)
#destroy_mlp_prediction(test_before)

train_classification_stochastic_gradient_backpropagation_mlp_model(p_model,
                                                                   inputs,
                                                                   outputs)

test_after = predict_mlp_model_classification(p_model, [0, 0])
print("test after:00", test_after)
#destroy_mlp_prediction(test_after)
test_after = predict_mlp_model_classification(p_model, [0, 1])
print("test after:01", test_after)
#destroy_mlp_prediction(test_after)
test_after = predict_mlp_model_classification(p_model, [1, 1])
print("test after:11", test_after)
#destroy_mlp_prediction(test_after)
test_after = predict_mlp_model_classification(p_model, [1, 0])
print("test after:10", test_after)
#destroy_mlp_prediction(test_after)

test_after = predict_mlp_model_classification(p_model, [-1, 0])
print("test after:-10", test_after)
#destroy_mlp_prediction(test_after)
test_after = predict_mlp_model_classification(p_model, [2, 0])
print("test after:21", test_after)
#destroy_mlp_prediction(test_after)

destroy_mlp_model(p_model)
"""
# """
import numpy as np
import os
from PIL import Image

IMG_SIZE = (8, 8)
PATH = os.path.join("data/")
TRAIN = os.path.join(PATH, "train")
classes = os.listdir(TRAIN)

def import_images_and_assign_labels(folder, label, X, Y):
    for file in os.listdir(folder):
        image_path = os.path.join(folder, file)
        im = Image.open(image_path)
        im = im.resize((8, 8))
        im = im.convert("RGB")
        im_arr = np.array(im)
        im_arr = np.reshape(im_arr, (8 * 8 * 3))
        X.append(im_arr)
        Y.append(label)


def import_dataset():
    X_train, y_train, X_valid, y_valid = [], [], [], []
    labels = np.identity(len(os.listdir(TRAIN)) - 1)

    for set_type in ["train", "valid"]:
        for cl, lab in zip(classes, labels):
            if set_type == "train":
                X_set, y_set = X_train, y_train
            else:
                X_set, y_set = X_valid, y_valid
            import_images_and_assign_labels(
                os.path.join(PATH, set_type, cl),
                lab,
                X_set,
                y_set
            )

    return (np.array(X_train) / 255.0, np.array(y_train)), \
           (np.array(X_valid) / 255.0, np.array(y_valid))

(X_train, y_train), (X_valid, y_valid) = import_dataset()

input_dim = [len(X_train[0]), 16, 8]

print(input_dim)

p_model, len_output_layer = create_mlp_model(input_dim)
test_before = predict_mlp_model_classification(p_model, X_train[0], len_output_layer)

print("Before training:", test_before)

train_classification_stochastic_gradient_backpropagation_mlp_model(p_model, X_train, y_train.flatten(), epochs=1000)

test_after = predict_mlp_model_classification(p_model, X_train[0], len_output_layer)

print("After training:", test_after)
# """
destroy_mlp_model(p_model)