import cv2
import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="model_stable_0.2.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

height, width = input_details[0]['shape'][1], input_details[0]['shape'][2]

def aiLayer(stream_img):
    global labels
    preditected_text = ""

    img = cv2.cvtColor(stream_img, cv2.COLOR_RGB2BGR)
    
    img_resized = cv2.resize(img, (width, height))
    img_normalized = (img_resized.astype(np.float32) / 127.5) - 1

    input_data = np.expand_dims(img_normalized, axis=0)

    if input_data.shape != tuple(input_details[0]['shape']):
        input_data = np.resize(input_data, input_details[0]['shape'])

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data)

    preditected_text = labels[predicted_class].split()[1]

    cv2.putText(img, f"Sinal: {labels[predicted_class]}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return img, preditected_text