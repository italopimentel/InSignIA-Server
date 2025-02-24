import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

interpreter = tf.lite.Interpreter(model_path="model_stable_0.2.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

#cap = cv2.VideoCapture(0)

def aiLayer(stream_img):

    img_rgb = cv2.cvtColor(stream_img, cv2.COLOR_BGR2RGB)
    
    h, w, _ = img_rgb.shape

    size = max(h, w)  
    square_img = np.zeros((size, size, 3), dtype=np.uint8)  
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    square_img[y_offset:y_offset+h, x_offset:x_offset+w] = img_rgb  # Centralizar a imagem

    
    square_img = cv2.resize(square_img, (224, 224))

    results = hands.process(square_img)
    preditected_text = ""
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(square_img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            input_data = square_img
            input_data = np.expand_dims(input_data, axis=0)
            input_data = (input_data / 255.0).astype(np.float32)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_class = np.argmax(output_data)
            preditected_text += chr(65 + predicted_class)

            cv2.putText(square_img, f"Letra: {chr(65 + predicted_class)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return square_img, preditected_text