from flask import Flask, request, jsonify, send_file
import os
import tempfile
from TTS.api import TTS
import whisper
from pydub import AudioSegment
import time
import requests

app = Flask(__name__)

load_start = time.time()
# Load the Whisper model
whisper_model = whisper.load_model("base")
print("Whisper model loaded successfully.")
    
    
# Initialize the TTS model
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
print("TTS model loaded successfully.")
load_end  = time.time()
#print(f"{INFO}: TIME TAKEN TO LOAD MODELS {(load_end-load_start)} seconds")
print("INFO: TIME TAKEN TO LOAD MODELS", (load_end-load_start), "seconds")

def chunk_text(text, max_length=250):
    """Helper function to split text into chunks of a maximum character length."""
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(" ".join(current_chunk + [word])) <= max_length:
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
    
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

@app.route('/')
def home():
    return "<h1>Welcome to the Audio Processing Backend</h1>"


@app.route('/process_audio', methods=['POST'])
def process_audio():
    """Endpoint to handle transcription and TTS generation from an uploaded audio file."""
    
    # Check if an audio file is provided in the request
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    
    audio_file = request.files['audio']

    try:
        # Save the uploaded audio file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
            tmp_audio_file.write(audio_file.read())
            tmp_audio_path = tmp_audio_file.name

        # Transcribe the audio using Whisper
        result = whisper_model.transcribe(tmp_audio_path)
        transcription_text = result['text'].strip()

        # Log the transcription for debugging
        print("Transcription from Whisper:", transcription_text)

        # Split transcription into manageable chunks
        text_chunks = chunk_text(transcription_text, max_length=250)

        # Initialize an empty AudioSegment object to concatenate all chunks
        final_audio = AudioSegment.silent(duration=0)

        # Process each chunk separately using TTS
        for idx, chunk in enumerate(text_chunks):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_part_{idx}.wav") as tmp_output_file:
                tts.tts_to_file(
                    text=chunk,
                    file_path=tmp_output_file.name,
                    speaker_wav=tmp_audio_path,  # Use uploaded audio for voice cloning
                    language="en"
                )

                # Load and append the generated chunk audio
                chunk_audio = AudioSegment.from_wav(tmp_output_file.name)
                final_audio += chunk_audio

        # Save the final concatenated audio to a new file
        combined_output_path = os.path.join(tempfile.gettempdir(), "combined_output.wav")
        final_audio.export(combined_output_path, format="wav")

        # Clean up temporary files
        os.remove(tmp_audio_path)

        # Prepare the response
        response_data = {
            "transcription": transcription_text,
            "generated_speech_url": request.host_url + 'download/' + os.path.basename(combined_output_path)
        }

        return jsonify(response_data), 200

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
        
@app.route('/download/<filename>', methods=['GET'])
def download_generated_audio(filename):
    """Endpoint to download the generated speech audio file."""
    file_path = os.path.join(tempfile.gettempdir(), filename)
    
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='audio/wav', as_attachment=True, download_name=filename)
    else:
        return jsonify({"error": "File not found"}), 404

# This will handle any cURL requests
@app.route('/curl_test', methods=['POST'])
def curl_test():
    """Test endpoint to ensure that the app can handle cURL requests."""
    data = request.form.get("data", None)
    if not data:
        return jsonify({"error": "No data provided in cURL request"}), 400
    
    # Just for demonstration, echo back the received data
    return jsonify({"received_data": data}), 200

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
#run using gunicorn -w 4 -k eventlet -b 0.0.0.0:5000 test_backend:app


