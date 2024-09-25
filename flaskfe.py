from flask import Flask, request, jsonify, send_file, render_template_string
import os
import tempfile
from TTS.api import TTS
import whisper
from pydub import AudioSegment
import time

app = Flask(__name__)

load_start = time.time()
# Load the Whisper model
whisper_model = whisper.load_model("base")
print("Whisper model loaded successfully.")

# Initialize the TTS model
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
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

# Route to Home Page with both file upload and audio recording functionality
@app.route('/')
def home():
    html_content = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Audio Transcription and Voice Cloning</title>
    </head>
    <body>
        <h1>Dysarthria Patients Transcription and Voice Cloning</h1>
        
        <!-- Audio Upload Form -->
        <h2>Upload an Audio File</h2>
        <form id="audioForm" method="POST" enctype="multipart/form-data" action="/process_audio">
            <label for="audio">Select an audio file (wav):</label>
            <input type="file" name="audio" id="audio" accept=".wav"><br><br>
            <button type="submit">Submit</button>
        </form>

        <!-- Audio Recording Controls -->
        <h2>Or Record Your Audio</h2>
        <button id="startRecord">Start Recording</button>
        <button id="stopRecord" disabled>Stop Recording</button>
        <audio id="audioPlayback" controls></audio>
        <button id="uploadRecorded" disabled>Upload Recorded Audio</button>
        
        <div id="result"></div>

        <script>
            const form = document.getElementById('audioForm');
            const resultDiv = document.getElementById('result');

            // Handle form submission for uploaded audio files
            form.addEventListener('submit', async function(event) {
                event.preventDefault();
                
                const formData = new FormData(form);
                resultDiv.innerHTML = 'Processing, please wait...';

                try {
                    const response = await fetch('/process_audio', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        throw new Error('Error: ' + response.statusText);
                    }

                    const data = await response.json();
                    
                    if (data.error) {
                        resultDiv.innerHTML = '<p style="color: red;">Error: ' + data.error + '</p>';
                    } else {
                        resultDiv.innerHTML = `
                            <h2>Transcription</h2>
                            <p>${data.transcription}</p>
                            <h2>Generated Speech</h2>
                            <a href="${data.generated_speech_url}" download>Download Generated Speech</a>
                        `;
                    }
                } catch (error) {
                    resultDiv.innerHTML = '<p style="color: red;">Error: ' + error.message + '</p>';
                }
            });

            // Recording audio functionality
            let mediaRecorder;
            let audioChunks = [];
            const startRecordButton = document.getElementById('startRecord');
            const stopRecordButton = document.getElementById('stopRecord');
            const audioPlayback = document.getElementById('audioPlayback');
            const uploadRecordedButton = document.getElementById('uploadRecorded');

            // Start recording
            startRecordButton.addEventListener('click', async () => {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);

                    mediaRecorder.start();
                    audioChunks = [];

                    mediaRecorder.ondataavailable = event => {
                        audioChunks.push(event.data);
                    };

                    mediaRecorder.onstop = () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        const audioUrl = URL.createObjectURL(audioBlob);
                        audioPlayback.src = audioUrl;
                        audioPlayback.style.display = 'block';
                        uploadRecordedButton.disabled = false;

                        // Upload the recorded audio when ready
                        uploadRecordedButton.addEventListener('click', async () => {
                            const formData = new FormData();
                            formData.append('audio', audioBlob, 'recorded_audio.wav');
                            resultDiv.innerHTML = 'Processing recorded audio, please wait...';

                            try {
                                const response = await fetch('/process_audio', {
                                    method: 'POST',
                                    body: formData
                                });

                                if (!response.ok) {
                                    throw new Error('Error: ' + response.statusText);
                                }

                                const data = await response.json();

                                if (data.error) {
                                    resultDiv.innerHTML = '<p style="color: red;">Error: ' + data.error + '</p>';
                                } else {
                                    resultDiv.innerHTML = `
                                        <h2>Transcription</h2>
                                        <p>${data.transcription}</p>
                                        <h2>Generated Speech</h2>
                                        <a href="${data.generated_speech_url}" download>Download Generated Speech</a>
                                    `;
                                }
                            } catch (error) {
                                resultDiv.innerHTML = '<p style="color: red;">Error: ' + error.message + '</p>';
                            }
                        });
                    };

                    startRecordButton.disabled = true;
                    stopRecordButton.disabled = false;
                } catch (error) {
                    console.error('Error accessing microphone:', error);
                }
            });

            // Stop recording
            stopRecordButton.addEventListener('click', () => {
                mediaRecorder.stop();
                startRecordButton.disabled = false;
                stopRecordButton.disabled = true;
            });
        </script>
    </body>
    </html>
    '''
    return render_template_string(html_content)

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """Combined endpoint for transcribing and generating speech using the same uploaded or recorded audio for cloning."""
    
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    
    audio_file = request.files['audio']

    try:
        # Save the uploaded or recorded audio file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
            tmp_audio_file.write(audio_file.read())
            tmp_audio_path = tmp_audio_file.name

        # Transcribe the audio using Whisper
        result = whisper_model.transcribe(tmp_audio_path)
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
                    speaker_wav=tmp_audio_path,  # Use the uploaded or recorded audio for voice cloning
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

