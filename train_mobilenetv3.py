import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import load_img, img_to_array

# Path to the dataset
data_dir = './content/data'

# Parameters
img_size = (224, 224)  # Image sizes for the model (default for MobileNetV3-Small is 224x224)
batch_size = 32
num_classes = 2  # Number of classes (four faces)

# Load the pre-trained MobileNetV3-Small model, without the top layers
base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))

# Add new top layers for our specific classification task
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # Adding a fully connected layer
predictions = Dense(num_classes, activation='softmax')(x)  # Output layer with softmax activation

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Make sure we can re-train the base model for fine-tuning
for layer in base_model.layers:
    layer.trainable = True

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Prepare the training data generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    rotation_range=20,
    brightness_range=(0.5, 1.5)
)

train_generator = train_datagen.flow_from_directory(
    directory= data_dir + "/train",
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
)

print("Classes:", train_generator.class_indices)

# Train the model
epochs = 1  # Number of epochs for training

model.fit(
    train_generator,
    epochs=epochs,
)

# Save the trained model
model.save('face_recognition_model.keras')

# Save the tflite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tfmodel = converter.convert()
open("face_recognition_model.tflite" , "wb") .write(tfmodel)

# Function to predict face from an image
def predict_face(img_path):
    img = load_img(img_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    print("Confidences:", predictions)
    predicted_class = np.argmax(predictions, axis=1)
    class_labels = list(train_generator.class_indices.keys())
    return class_labels[predicted_class[0]]

# Example prediction
img_path = './content/data/train/Pierce/Pierce14.jpeg'  # File path to the test image
predicted_face = predict_face(img_path)
print(f'Predicted face: {predicted_face}')