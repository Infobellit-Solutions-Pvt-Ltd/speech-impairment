from flask import Flask, request, jsonify, send_file
import os
import tempfile
from TTS.api import TTS
import whisper
from pydub import AudioSegment
import time 

app = Flask(__name__)

load_start = time.time()
# Load the Whisper model
whisper_model = whisper.load_model("large-v3")
print("Whisper model loaded successfully.")

# Initialize the TTS model
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
print("TTS model loaded successfully.")
load_end  = time.time()
print("{INFO}: TIME TAKEN TO LOAD MODELS", (load_end-load_start))

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
    
    
#route to Home Page
@app.route('/')
def hello_world():
    return "<h1>Hey there! <br> backendend script running here!!</h1>"
    
@app.route('/process_audio', methods=['POST'])
def process_audio():
    """Combined endpoint for transcribing and generating speech using the same uploaded audio for cloning."""
    
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    
    audio_file = request.files['audio']

    try:
        # Save the uploaded audio file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
            tmp_audio_file.write(audio_file.read())
            tmp_audio_path = tmp_audio_file.name

        # Transcribe the audio using Whisper
        result = whisper_model.transcribe(tmp_audio_path,language='en')
        transcription_text = result['text'].strip()

        # Print the transcription before sending to TTS
        print("Transcription from Whisper:", transcription_text)

        # Split the transcription into chunks if it exceeds the character limit
        text_chunks = chunk_text(transcription_text, max_length=250)

        # Create an empty AudioSegment object to concatenate all chunks
        final_audio = AudioSegment.silent(duration=0)

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

        # Save the final concatenated audio to a new file
        combined_output_path = os.path.join(tempfile.gettempdir(), "combined_output.wav")
        final_audio.export(combined_output_path, format="wav")

        # Clean up temporary files
        os.remove(tmp_audio_path)

        # Return the transcription and the generated speech file
        return jsonify({
            "transcription": transcription_text,
            "generated_speech_url": request.host_url + 'download/' + os.path.basename(combined_output_path)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_generated_audio(filename):
    """Endpoint to download the generated speech audio file."""
    file_path = os.path.join(tempfile.gettempdir(), filename)
    
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='audio/wav', as_attachment=True, download_name=filename)
    else:
        return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
#curl -X POST http://localhost:5000/process_audio -F "audio=@./sample.mp3"
