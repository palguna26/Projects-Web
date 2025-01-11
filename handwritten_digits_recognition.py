import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def load_mnist_data():
    """Load and normalize the MNIST dataset."""
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)
    return X_train, y_train, X_test, y_test


def create_model():
    """Create a neural network model for digit recognition."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(model, X_train, y_train, X_test, y_test, epochs=3):
    """Train the model and plot training history."""
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))

    # Plot training and validation accuracy0
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(f"Validation Loss: {val_loss}")
    print(f"Validation Accuracy: {val_acc}")
    return model


def load_or_train_model(train_new_model=True, epochs=3):
    """Load an existing model or train a new one."""
    if train_new_model:
        X_train, y_train, X_test, y_test = load_mnist_data()
        model = create_model()
        model = train_model(model, X_train, y_train, X_test, y_test, epochs)
        model.save('handwritten_digits.h5')
        print("Model trained and saved as 'handwritten_digits.h5'.")
    else:
        if os.path.isfile('handwritten_digits.h5'):
            model = tf.keras.models.load_model('handwritten_digits.h5')
            print("Model loaded from 'handwritten_digits.h5'.")
        else:
            print("No saved model found. Please train a new model.")
            exit()
    return model


def predict_custom_images(model, image_dir='digits/', retrain_dir='retrain/'):
    """Predict digits from custom images in the specified directory."""
    if not os.path.exists(image_dir):
        print(f"Directory '{image_dir}' not found. Please add images and try again.")
        return

    if not os.path.exists(retrain_dir):
        os.makedirs(retrain_dir)

    image_number = 1
    while os.path.isfile(f'{image_dir}digit{image_number}.png'):
        file_name = f'digit{image_number}.png'
        file_path = f'{image_dir}{file_name}'
        try:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (28, 28))
            img_resized = np.invert(np.array([img_resized]))  # Invert the image for better prediction
            img_resized = img_resized.astype('float32') / 255.0  # Normalize the image

            prediction = model.predict(img_resized)
            predicted_digit = np.argmax(prediction)
            print(f"File: {file_name} | Predicted: {predicted_digit}")

            # Show the image
            plt.imshow(img_resized[0], cmap=plt.cm.binary)
            plt.title(f"File: {file_name} | Predicted: {predicted_digit}")
            plt.show()

            # Ask user for feedback
            user_feedback = input(f"Is the prediction for '{file_name}' correct? (yes/no): ").strip().lower()
            if user_feedback == 'no':
                correct_label = input("What is the correct digit? (0-9): ").strip()
                if correct_label.isdigit() and 0 <= int(correct_label) <= 9:
                    correct_label = int(correct_label)
                    # Save image and label for retraining
                    save_path = os.path.join(retrain_dir, f"{file_name.split('.')[0]}_label{correct_label}.png")
                    cv2.imwrite(save_path, img)
                    print(f"Saved for retraining as {save_path}.")
                else:
                    print("Invalid input. Skipping this image.")
        except Exception as e:
            print(f"Error reading image '{file_name}': {e}")
        finally:
            image_number += 1


if __name__ == "__main__":
    print("Welcome to Handwritten Digits Recognition!")

    # Choose whether to train a new model or load an existing one
    train_new_model = input("Do you want to train a new model? (yes/no): ").strip().lower() == 'yes'
    epochs = 3  # Default epochs, you can make this user-configurable

    # Load or train the model
    model = load_or_train_model(train_new_model, epochs)

    # Predict custom images and gather feedback
    predict_custom_images(model)
