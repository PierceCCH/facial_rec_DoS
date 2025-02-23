import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('face_recognition_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tfmodel = converter.convert()
open ("face_recognition_model.tflite" , "wb") .write(tfmodel)