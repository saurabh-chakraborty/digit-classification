
# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import metrics, svm
from sklearn.model_selection import train_test_split


# Flatten the images
def preprocess_data(data):
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data

# Split data into train and test subsets
def split_data(x, y, test_size, random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Train a model of choice, pass the model parameters
def train_model(x, y, model_params, model_type="svm"):
    if (model_type=="svm"):
        # Create a SVM classifier
        clf = svm.SVC
    # Create a model object
    model = clf(**model_params)
    # Training model
    model.fit(x, y)
    return model

# Predict & Eval
def predict_and_eval(model, x, y):
    # Predict the value of the digit on the test subset
    predicted = model.predict(x)

    # Below we visualize the first 4 test samples and show their predicted digit value in the title.
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, x, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

    # :func:`~sklearn.metrics.classification_report` builds a text report showing the main classification metrics.

    print(
        f"Classification report for classifier {model}:\n"
        f"{metrics.classification_report(y, predicted)}\n"
    )

    return predicted

# Reporting Metrics
def report_CM(y, predicted):
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    plt.show()

    # The ground truth and predicted lists
    y_true = []
    y_pred = []
    cm = disp.confusion_matrix

    # For each cell in the confusion matrix, add the corresponding ground truths and predictions to the lists
    for gt in range(len(cm)):
        for pred in range(len(cm)):
            y_true += [gt] * cm[gt][pred]
            y_pred += [pred] * cm[gt][pred]

    print(
        "Classification report rebuilt from confusion matrix:\n"
        f"{metrics.classification_report(y_true, y_pred)}\n"
    )