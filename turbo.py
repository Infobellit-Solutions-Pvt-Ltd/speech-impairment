import os
import tempfile
import whisper
import time
import psutil
from flask_cors import CORS
from flask import Flask, request, jsonify, send_file
from flask_socketio import SocketIO, emit
from TTS.api import TTS
from pydub import AudioSegment
#=============================================================================================
app = Flask(__name__, static_folder="./build", static_url_path="/")
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
#=============================================================================================

# Track load times
whisper_load_time = 0
tts_load_time = 0
total_load_time = 0
#=============================================================================================

# Measure Whisper model load time
load_start_whisper = time.time()
whisper_model = whisper.load_model("large-v3")
whisper_load_time = time.time() - load_start_whisper
print(f"Whisper model loaded successfully in {whisper_load_time:.2f} seconds.")

# Measure TTS model load time
load_start_tts = time.time()
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
tts_load_time = time.time() - load_start_tts
print(f"TTS model loaded successfully in {tts_load_time:.2f} seconds.")

# Calculate total model load time
total_load_time = whisper_load_time + tts_load_time
print(f"Total time taken to load models: {total_load_time:.2f} seconds.")
#=============================================================================================

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
#=============================================================================================

# Route to Home Page
@app.route('/')
def hello_world():
    return "<h1>Hey there! <br> Backend script running here!!</h1>"

# # Route to Home Page
# @app.route('/ui')
# def hello_world2():
#      return send_from_directory(app.static_folder, 'index.html')
#=============================================================================================

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """Combined endpoint for transcribing and generating speech using the same uploaded audio for cloning."""
    
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    
    audio_file = request.files['audio']

    try:
        # Measure total response time
        response_start_time = time.time()

        # Save the uploaded audio file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
            tmp_audio_file.write(audio_file.read())
            tmp_audio_path = tmp_audio_file.name

        # Measure transcription time (Whisper)
        transcription_start_time = time.time()
        result = whisper_model.transcribe(tmp_audio_path, language='en')
        transcription_text = result['text'].strip()
        transcription_time = time.time() - transcription_start_time
        print(f"Transcription completed in {transcription_time:.2f} seconds.")

        # Print the transcription before sending to TTS
        print("Transcription from Whisper:", transcription_text)

        # Split the transcription into chunks if it exceeds the character limit
        text_chunks = chunk_text(transcription_text, max_length=250)

        # Create an empty AudioSegment object to concatenate all chunks
        final_audio = AudioSegment.silent(duration=0)

        # Measure TTS generation time
        tts_start_time = time.time()

        # Process each chunk separately using TTS
        for idx, chunk in enumerate(text_chunks):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_part_{idx}.wav") as tmp_output_file:
                tts.tts_to_file(
                    text=chunk,
                    file_path=tmp_output_file.name,
                    speaker_wav=tmp_audio_path,  # Use the uploaded audio for voice cloning
                    language="en"
                )
                
                # Load the generated chunk audio and append it to the final audio
                chunk_audio = AudioSegment.from_wav(tmp_output_file.name)
                final_audio += chunk_audio

        tts_generation_time = time.time() - tts_start_time
        print(f"TTS generation completed in {tts_generation_time:.2f} seconds.")

        # Save the final concatenated audio to a new file
        combined_output_path = os.path.join(tempfile.gettempdir(), "combined_output.wav")
        final_audio.export(combined_output_path, format="wav")

        # Clean up temporary files
        os.remove(tmp_audio_path)

        # Measure total response time
        total_response_time = time.time() - response_start_time
        print(f"Total time taken to generate response: {total_response_time:.2f} seconds.")

        # Return the transcription and the generated speech file
        return jsonify({
            "transcription": transcription_text,
            "transcription_time": f"{transcription_time:.2f} seconds",
            "tts_generation_time": f"{tts_generation_time:.2f} seconds",
            "total_response_time": f"{total_response_time:.2f} seconds",
            "generated_speech_url": request.host_url + 'download/' + os.path.basename(combined_output_path)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
#=============================================================================================

@app.route('/download/<filename>', methods=['GET'])
def download_generated_audio(filename):
    """Endpoint to download the generated speech audio file."""
    file_path = os.path.join(tempfile.gettempdir(), filename)
    
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='audio/wav', as_attachment=True, download_name=filename)
    else:
        return jsonify({"error": "File not found"}), 404
#=============================================================================================

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5050)

# if __name__ == '__main__':
#     # Specify the paths to your SSL certificate and key files
#     ssl_cert_path = 'ssl_cert.pem'
#     ssl_key_path = 'ssl_key.pem'

#     # Run Flask with SSL enabled
#     app.run(debug=False, host='0.0.0.0', port=5050, ssl_context=(ssl_cert_path, ssl_key_path))