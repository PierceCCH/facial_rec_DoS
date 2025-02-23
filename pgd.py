import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

from PIL import Image

class_labels = ['Allen', 'Ethan', 'Nhat', 'Zaya']

def predict_face_attack(img, model):
    img = tf.expand_dims(img, 0)
    predictions = model(img)

    predicted_class = tf.argmax(predictions, 1)
    return predictions

def pgd_attack(model, x, y, eps=0.3, alpha=0.007, iters=20):
    x_adv = tf.Variable(initial_value=x/255, trainable=True) # Initialize adversarial example as a copy of the input
    
    for i in range(iters):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x_adv)
            prediction = predict_face_attack(x_adv, model)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, prediction)

        gradient = tape.gradient(loss, x_adv)
        x_adv = x_adv + alpha * tf.sign(gradient)
        x_adv = tf.clip_by_value(x_adv, x/255 - eps, x/255 + eps)
        x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)
    
    x_adv = x_adv*255
    return x_adv

def save_mask(model, class_label):
    img_path = './content/data/train/'+ class_labels[class_label] + '/' + class_labels[class_label] + '300.jpeg'
    save_path = './masks/' + class_labels[class_label] + '_mask.jpeg'
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    mask = pgd_attack(model, img_tensor, class_label) - img_tensor
    mask = mask.numpy().astype(np.uint8)
    im = Image.fromarray(mask)
    im.save(save_path)
    print('Mask for ' + class_labels[class_label] + ': Done')
    return mask

def main():
    model = load_model('face_recognition_model_final.h5')
    for i in range(4):
        save_mask(model, tf.convert_to_tensor(i))

if __name__ == '__main__':
    main()