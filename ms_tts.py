import os
import tempfile
import whisper
import time
import torch
import soundfile as sf
from flask_cors import CORS
from flask import Flask, request, jsonify, send_file
from flask_socketio import SocketIO, emit
from pydub import AudioSegment
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

#=============================================================================================
app = Flask(__name__, static_folder="./build", static_url_path="/")
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load models on CPU
whisper_model = whisper.load_model("base.en", device="cpu")
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to("cpu")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to("cpu")

# Default speaker embedding
default_embedding = torch.zeros(1, 512)

#=============================================================================================
def split_text_into_chunks(text, max_tokens=600):
    """Split the text into chunks so each fits within the token limit."""
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(processor.tokenizer(" ".join(current_chunk + [word]))['input_ids']) <= max_tokens:
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

@app.route('/')
def hello_world():
    return "<h1>Hey there! <br> Backend script running here!!</h1>"

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    
    audio_file = request.files['audio']

    try:
        response_start_time = time.time()

        # Save uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
            tmp_audio_file.write(audio_file.read())
            tmp_audio_path = tmp_audio_file.name

        # Transcription with Whisper
        transcription_start_time = time.time()
        result = whisper_model.transcribe(tmp_audio_path, language='en')
        transcription_text = result['text'].strip()
        transcription_time = time.time() - transcription_start_time
        print(f"Transcription completed in {transcription_time:.2f} seconds.")

        # Split transcription into chunks
        text_chunks = split_text_into_chunks(transcription_text)

        # Concatenate audio chunks
        final_audio = AudioSegment.silent(duration=0)

        # Generate speech for each chunk
        tts_start_time = time.time()
        for idx, chunk in enumerate(text_chunks):
            inputs = processor(text=chunk, return_tensors="pt")
            speech = model.generate_speech(inputs["input_ids"], default_embedding, vocoder=vocoder)

            # Save each chunk temporarily and concatenate
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_part_{idx}.wav") as tmp_output_file:
                sf.write(tmp_output_file.name, speech.numpy(), samplerate=16000)
                chunk_audio = AudioSegment.from_wav(tmp_output_file.name)
                final_audio += chunk_audio

        tts_generation_time = time.time() - tts_start_time
        print(f"TTS generation completed in {tts_generation_time:.2f} seconds.")

        # Save final audio
        output_path = os.path.join(tempfile.gettempdir(), "generated_speech.wav")
        final_audio.export(output_path, format="wav")

        os.remove(tmp_audio_path)  # Clean up

        total_response_time = time.time() - response_start_time
        print(f"Total response time: {total_response_time:.2f} seconds.")

        return jsonify({
            "transcription": transcription_text,
            "transcription_time": f"{transcription_time:.2f} seconds",
            "tts_generation_time": f"{tts_generation_time:.2f} seconds",
            "total_response_time": f"{total_response_time:.2f} seconds",
            "generated_speech_url": request.host_url + 'download/' + os.path.basename(output_path)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_generated_audio(filename):
    file_path = os.path.join(tempfile.gettempdir(), filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='audio/wav', as_attachment=True, download_name=filename)
    else:
        return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5050)
