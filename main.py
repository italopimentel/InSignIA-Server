from flask import Flask, request, Response
import cv2
import numpy as np
import queue
import threading
from ailayer import aiLayer

app = Flask(__name__)

last_readed_text = ""
current_readed_text = ""
app_should_read = False
frame_queue = queue.Queue(maxsize=40) 
lock = threading.Lock()

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'frame' not in request.files:
        print("Nenhuma imagem recebida!")
        return "Nenhuma imagem recebida", 400

    frame_data = request.files['frame'].read()

    if len(frame_data) == 0:
        print("Arquivo vazio!")
        return "Arquivo vazio", 400

    print(f"Recebendo frame de {len(frame_data)} bytes")

    frame_array = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

    if frame is None:
        print("Erro ao decodificar a imagem!")
        return "Erro ao decodificar a imagem", 400

    
    with lock:
        if frame_queue.full():
            frame_queue.get()
        frame_queue.put(frame)

    return "Quadro recebido", 200

last_posted_text = ""
@app.route('/get-result', methods=['GET'])
def display_result():
    global app_should_read, last_posted_text
    return {"should_read" : app_should_read,
            "last_read" : last_posted_text
            }

"""
should_process = True 
def process_image():
    global frame_queue, display_active, last_readed_text, current_readed_text, app_should_read
    while should_process:
        if not frame_queue.empty():
            processed_frame, predicted_text = aiLayer(frame_queue.get())
            if (predicted_text != "" and predicted_text != last_readed_text):
                app_should_read = True
                last_readed_text = predicted_text
                current_readed_text = predicted_text
            else: 
                app_should_read = False
"""


display_active = True 
sucessful_frame_count = 0
def display_frames():
    global frame_queue, display_active, last_readed_text, current_readed_text, app_should_read,sucessful_frame_count,last_posted_text
    while display_active:
        if not frame_queue.empty():
            processed_frame, predicted_text = aiLayer(frame_queue.get())
            if (predicted_text != None):
                last_readed_text = current_readed_text
                current_readed_text = predicted_text
                if current_readed_text == last_readed_text:
                    sucessful_frame_count +=1
                else:
                    sucessful_frame_count = 0
                
                if (sucessful_frame_count >= 3 and current_readed_text != last_posted_text):
                    app_should_read = True
                    sucessful_frame_count = 0
                    last_posted_text = current_readed_text 
                else:
                    app_should_read = False
            else: 
                app_should_read = False
            cv2.imshow("Frame Recebido", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                display_active = False
                break
    cv2.destroyAllWindows()


display_thread = threading.Thread(target=display_frames, daemon=True)
display_thread.start()

@app.route('/')
def index():
    return "Bem vindo ao InSignea app"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
    display_active = False  
    display_thread.join()
