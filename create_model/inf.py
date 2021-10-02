import numpy as np
import tensorflow as tf
import cv2

img = cv2.imread("C://Users//Admin//Downloads//metal2.jpg")
img = cv2.resize(img, (224,224))
img = np.array(img, dtype="float32")
img = np.reshape(img, (1,224,224,3))


# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']

print("*"*50, input_details)
interpreter.set_tensor(input_details[0]['index'], img)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
class_names = ['battery', 'biological', 'glass', 'cardboard', 'clothes', 'metal', 'paper', 'plastic', 'trash']
# class_names = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 
#                'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

# class_names = ['tao', 'chuoi', 'cam', 'tao hong', 'chuoi hong', 'cam hong']

print(
    "Hình ảnh này rất có thể là '{}' với {:.2f} %."
    .format(class_names[np.argmax(output_data)], 100 * np.max(output_data))
)
# print(output_data)