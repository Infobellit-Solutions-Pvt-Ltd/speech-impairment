from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import whisper
import io
import os
import base64
import numpy as np
from scipy.io import wavfile
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  
socketio = SocketIO(app, cors_allowed_origins="*")

# Load Whisper base/tiny model
model = whisper.load_model("base")
print("Whisperloded successfully.")

# Set upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route to handle file uploads
@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    #transcribe the file and return the transcription
    result = model.transcribe(file_path)
    transcription = result['text']
    
    return jsonify({'transcription': transcription, 'message': 'File uploaded successfully'}), 200

# Route to handle live transcription via WebSocket
@socketio.on('audio_stream')
def handle_audio_stream(data):
    try:
        if 'data' not in data:
            raise ValueError("No data field in message")
        
        # Decode base64 data
        audio_data = base64.b64decode(data['data'])
        audio_stream = io.BytesIO(audio_data)
        
        # Create a temporary file for the audio stream
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        # Transcribe the temporary file
        result = model.transcribe(temp_file_path)
        transcription = result['text']
        os.remove(temp_file_path)

        emit('transcription_update', {'transcription': transcription})
    except Exception as e:
        print(f"Error processing audio stream: {e}")

@app.route('/')
def hello_world():
    return "<h1>Hey hi! this is backendend script running here dude!!</h1>"

if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
#run using gunicorn -w 4 -k eventlet -b 0.0.0.0:5000 app:app
