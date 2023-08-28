import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

# Load your handcrafted feature dataset, X, and target labels, y
feature_matrix = pd.read_csv('/content/Combined.csv')
X = feature_matrix.iloc[:, :-1].values
y = feature_matrix['label'].values

encoder = LabLabelEncoderLabLabelEncoderelEncoderelEncoder()
y = encoder.fit_transform(y)
y = np_utils.to_categorical(y)  # convert to one-hot encoding

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the feature data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the ANN model
model = Sequential()
model.add(Dense(units=64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=y_train.shape[1], activation='softmax'))

# Compile the model with your choice of loss function, optimizer, and metrics
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the ANN model on the training set
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# Use the trained ANN model to predict on the testing set
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# Evaluate the accuracy of the ANN model on the testing set
accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred)
precision = precision_score(np.argmax(y_test, axis=1), y_pred, average = 'weighted')
f1 = f1_score(np.argmax(y_test, axis=1), y_pred, average = 'weighted')

cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
num_samples_per_class = np.sum(cm, axis=1)
cm_percentage = np.zeros_like(cm, dtype=float)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        cm_percentage[i, j] = cm[i, j] / num_samples_per_class[i] if num_samples_per_class[i] > 0 else 0


# Create a heatmap of the confusion matrix
sns.heatmap(cm_percentage, annot=True, cmap="Blues", fmt=".3f", cbar=False)

# Add labels and title to the plot
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix for ANN")

# Show the plot
plt.show()



print("Accuracy: {:.4f}".format(accuracy))
print("Precision: {:.4f}".format(precision))
print("F1 Score: {:.4f}".format(f1))