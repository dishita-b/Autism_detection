import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



"""# Feature Extraction of each image from dataset and store as dataframe"""

import os
import cv2
import matplotlib.pyplot as plt

# For deep learning feature extraction
from tensorflow.keras.applications import VGG16, ResNet101, MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess

from sklearn.manifold import TSNE
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix

dataset_dir = "/kaggle/input/dataset-autistic/processed_faces"
class_names = ['Non-Autistic', 'Autistic']

features_list = []
labels = []

for label, class_name in enumerate(class_names):
    print(label,"->", class_name)

def get_pretrained_model(model_name):
    """
    Returns a pre-trained model without the top layers and with a GlobalAveragePooling layer.
    """
    if model_name == "VGG16":
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        preprocess = vgg_preprocess
    elif model_name == "ResNet101":
        base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        preprocess = resnet_preprocess
    elif model_name == "MobileNet":
        base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        preprocess = mobilenet_preprocess
    else:
        raise ValueError("Unsupported model name.")

    # Add global average pooling to convert feature maps into a flat vector.
    x = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=x)
    return model, preprocess

def extract_features_from_image(image, model_name):
    """
    Resizes the image, preprocesses it according to the model,
    and returns the feature vector extracted by the CNN.
    """
    model, preprocess = get_pretrained_model(model_name)
    # Resize image to the target size (224x224 for these models)
    image_resized = cv2.resize(image, (224, 224))
    image_array = np.expand_dims(image_resized, axis=0)
    image_preprocessed = preprocess(image_array)
    features = model.predict(image_preprocessed)
    return features.flatten()

for label, class_name in enumerate(class_names):
    class_folder = os.path.join(dataset_dir, class_name)
    for filename in os.listdir(class_folder):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(class_folder, filename)
            image = cv2.imread(img_path)
            # norm_img = image.astype("float32") / 255.0
            feat = extract_features_from_image(image, model_name="VGG16")
            features_list.append(feat)
            labels.append(label)

# for label, class_name in enumerate(class_names):
#     class_folder = os.path.join(dataset_dir, class_name)
#     for i, filename in enumerate(os.listdir(class_folder)):
#         if i >= 20:
#             break
#         if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
#             img_path = os.path.join(class_folder, filename)
#             image = cv2.imread(img_path)
#             # norm_img = image.astype("float32") / 255.0
#             feat = extract_features_from_image(image, model_name="VGG16")
#             features_list.append(feat)
#             labels.append(label)

len(features_list)

df_features = pd.DataFrame(features_list)
df_features.to_csv("VGG16_features.csv", index=False)

df_labels = pd.DataFrame(labels)
df_labels.to_csv("image_labels.csv", index=False)

# df_loaded_features = pd.read_csv("VGG16_features.csv")
# df_loaded_label = pd.read_csv("image_labels.csv")
# X = df_loaded_features.to_numpy()
# y = df_loaded_label.to_numpy()

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

# # Assume 'features' is a numpy array of shape (n_samples, n_features)
# # For example, you might load it from a file or obtain it after processing images.
# # features = np.load('features.npy')

# # --------------------------
# # Step 1: Analyze Explained Variance
# # --------------------------
# # Fit PCA without reducing dimensions to get the full explained variance ratio.
# pca_full = PCA()
# pca_full.fit(features)

# # Compute cumulative explained variance
# cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# # Plot the cumulative explained variance to visualize the number of components needed
# plt.figure(figsize=(8, 6))
# plt.plot(cumulative_variance, marker='o')
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.title('Cumulative Explained Variance by PCA Components')
# plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance Threshold')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Decide the number of components required to retain 95% of the variance.
# n_components = np.argmax(cumulative_variance >= 0.95) + 1
# print("Number of components to retain 95% variance:", n_components)

# # --------------------------
# # Step 2: Apply PCA with Whitening
# # --------------------------
# # Whitening scales the components to have unit variance and can help some classifiers.
# pca = PCA(n_components=n_components, whiten=True, random_state=42)
# features_reduced = pca.fit_transform(features)

# print("Original feature shape:", features.shape)
# print("Reduced feature shape:", features_reduced.shape)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------
# Load features and labels from CSV files
# --------------------------
df_loaded_features = pd.read_csv("VGG16_features.csv")
df_loaded_label = pd.read_csv("image_labels.csv")
X = df_loaded_features.to_numpy()  # Feature matrix
y = df_loaded_label.to_numpy()     # Labels

print("Original feature shape:", X.shape)
print("Labels shape:", y.shape)

# --------------------------
# Step 3: Train-Test Split
# --------------------------
# Split the reduced features and labels into training and test sets.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)

print("Training features shape:", X_train.shape)
print("Test features shape:", X_test.shape)
print("Training labels shape:", y_train.shape)
print("Test labels shape:", y_test.shape)

# pip install openTSNE

# from openTSNE import TSNE
# def reduce_features_tsne(features, n_components):
#     """
#     Reduces the high-dimensional features to n_components using t-SNE.
#     """
#     tsne = TSNE(n_components=n_components, perplexity=30, n_jobs=-1)
#     reduced_features = tsne.fit(features)
#     return reduced_features

# from sklearn.decomposition import PCA
# pca = PCA(n_components=200, random_state=42)
# pca_features = pca.fit_transform(X)

# X_reduced = reduce_features_tsne(X, n_components=35)

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def train_xgboost_classifier(X_train, y_train):
    """
    Trains an XGBoost classifier on the provided features.
    """
    model = xgb.XGBClassifier(n_estimators = 1000,learning_rate=0.08,max_depth=10, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

def train_random_forest_classifier(X_train, y_train):
    """
    Trains a Random Forest classifier on the provided features.
    """
    clf = RandomForestClassifier(n_estimators=10000,max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the classifier and prints performance metrics.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    precision = precision_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    print(f"Accuracy: {acc*100:.2f}%")
    print(f"AUC: {auc*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Sensitivity (Recall): {sensitivity*100:.2f}%")
    print(f"Specificity: {specificity*100:.2f}%")
    return cm

xgb_model = train_xgboost_classifier(X_train, y_train)
print("XGBoost Classifier Performance:")
cm_xgb = evaluate_model(xgb_model, X_test, y_test)

xgb_model.save_model("VGG16_xgb.json")  # Save in JSON format

import xgboost as xgb
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("VGG16_xgb.json")  # Load the model

rf_model = train_random_forest_classifier(X_train, y_train)
print("\nRandom Forest Classifier Performance:")
cm_rf = evaluate_model(rf_model, X_test, y_test)

