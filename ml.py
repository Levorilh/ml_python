from linear import *from mlp import *from rbfn import *init_random()"""import matplotlib.pyplot as pltimport numpy as npX = np.array([      [2.5, 4],      [2, 3],      [3, 3]])Y = np.array([      1,      -1,      -1])num_classes = 2k = 2input_dim = 2expected_output = [[1,0] if label >= 0 else [0,1] for label in Y]model = create_rbfn_model(input_dim, num_classes, k)train_rbfn_model(model, X, expected_output)colors = ['blue' if coord >= 0 else 'red' for coord in Y]plot_input = [[x,y] for x in range(6) for y in range(6)]plot_output_colors = ['blue' if predict[0] > predict[1] else 'red' for predict in [predict_rbfn(model, coord) for coord in plot_input]]plt.scatter([p[0] for p in X], [p[1] for p in X], c=colors, s=200)plt.scatter([p[0] for p in plot_input], [p[1] for p in plot_input], c=plot_output_colors)plt.show()path = "C:\\Users\\N\\Desktop\\test_rbf_model.txt";save_rbf_model(model,path)model_load = load_rbf_model(path)plot_input = [[x,y] for x in range(6) for y in range(6)]plot_output_colors = ['blue' if predict[0] > predict[1] else 'red' for predict in [predict_rbfn(model_load, coord) for coord in plot_input]]plt.scatter([p[0] for p in plot_input], [p[1] for p in plot_input], c=plot_output_colors)plt.show()X = np.array([[-3., 3.], [-3., 2.], [-4., 2.], [-4., 3.], [-3.5, 3.5], [4., 1.], [5., 1.], [4., 0.5], [5., 0.5], [4.5, 1.5]      #[2.5, 4],[2, 3],[3, 3]])Y = np.array([[1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.], [0., 1.], [0., 1.], [0., 1.], [0., 1.]      #1,-1,-1])num_classes = 2k = 2input_dim = 2expected_output = [y for y in Y]    #[[1,0] if label >= 0 else [0,1] for label in Y]model = create_rbfn_model(input_dim, num_classes, k)train_rbfn_model(model, X, expected_output)predict = predict_rbfn(model, X[1])import numpy as npimport osfrom PIL import Imageimport matplotlib.pyplot as pltIMG_SIZE = (8, 8)PATH = os.path.join("data_large/")TRAIN = os.path.join(PATH, "train")classes = os.listdir(TRAIN)def import_images_and_assign_labels(folder, label, X, Y):    for file in os.listdir(folder):        image_path = os.path.join(folder, file)        im = Image.open(image_path)        im = im.resize(IMG_SIZE)        im = im.convert("RGB")        im_arr = np.array(im)        im_arr = np.reshape(im_arr, (IMG_SIZE[0] * IMG_SIZE[1] * 3,))        X.append(im_arr)        Y.append(label)def import_dataset():    X_train, y_train, X_valid, y_valid = [], [], [], []    labels = np.identity(len(os.listdir(TRAIN)))    for set_type in ["train", "valid"]:        for cl, lab in zip(classes, labels):            if set_type == "train":                X_set, y_set = X_train, y_train            else:                X_set, y_set = X_valid, y_valid            import_images_and_assign_labels(                os.path.join(PATH, set_type, cl),                lab,                X_set,                y_set            )    return (np.array(X_train) / 255.0, np.array(y_train)), \           (np.array(X_valid) / 255.0, np.array(y_valid))def accuracy(model):    true_preds = 0    total_preds = len(X_train)    for x, y in zip(X_train, y_train):        if np.argmax(predict_mlp_model_classification(model, x, input_dim[-1])) == np.argmax(y):            true_preds += 1    print(f"Accuracy training: {round((true_preds / total_preds) * 100, 2)}%")    true_preds = 0    total_preds = len(X_valid)    for x, y in zip(X_valid, y_valid):        if np.argmax(predict_mlp_model_classification(model, x, input_dim[-1])) == np.argmax(y):            true_preds += 1    print(f"Accuracy valid: {round((true_preds / total_preds) * 100, 2)}%")(X_train, y_train), (X_valid, y_valid) = import_dataset()picture_test = np.random.randint(0, len(X_valid) - 1)input_dim = [len(X_train[0]), 128, 8]p_model2 = load_mlp_model("models/mlp/MLP_50000_8x8_8_t_acc-34.56_v_acc-32.36.txt")test_after = predict_mlp_model_classification(p_model2, X_valid[picture_test], input_dim[-1])print("After training:", test_after)print("Class index : ", np.argmax(test_after))print("Class expected :", np.argmax(y_train[picture_test]))accuracy(p_model2)destroy_mlp_model(p_model2)"""