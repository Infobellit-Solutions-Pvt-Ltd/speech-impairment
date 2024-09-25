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
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                background-color: #f0f2f5;
            }

            h1, h2 {
                color: #333;
            }

            h1 {
                text-align: center;
                margin-bottom: 40px;
            }

            form, .record-section {
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
            }

            .container {
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }

            label {
                font-weight: bold;
                margin-bottom: 5px;
                display: block;
            }

            input[type="file"] {
                margin-bottom: 20px;
            }

            button {
                background-color: #007bff;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }

            button:hover {
                background-color: #0056b3;
            }

            audio {
                margin-top: 20px;
                width: 100%;
            }

            #result {
                margin-top: 20px;
                padding: 10px;
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 5px;
                display: none;
            }

            #result a {
                color: #007bff;
                text-decoration: none;
            }

            #result a:hover {
                text-decoration: underline;
            }

            .button-group {
                display: flex;
                gap: 10px;
                margin-top: 20px;
            }

            #startRecord, #stopRecord, #uploadRecorded {
                display: inline-block;
                margin: 10px 0;
            }

            #stopRecord, #uploadRecorded {
                display: none;
            }

            /* Hide the player initially */
            #uploadedAudioPlayback {
                display: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Audio Transcription and Voice Cloning</h1>

            <!-- Audio Upload Form -->
            <form id="audioForm" method="POST" enctype="multipart/form-data" action="/process_audio">
                <h2>Upload an Audio File</h2>
                <label for="audio">Select an audio file (wav):</label>
                <input type="file" name="audio" id="audio" accept=".wav" required><br><br>
                <audio id="uploadedAudioPlayback" controls></audio><br><br>
                <button type="submit">Submit</button>
            </form>

            <!-- Audio Recording Controls -->
            <div class="record-section">
                <h2>Or Record Your Audio</h2>
                <div class="button-group">
                    <button id="startRecord">Start Recording</button>
                    <button id="stopRecord">Stop Recording</button>
                    <button id="uploadRecorded">Upload Recorded Audio</button>
                </div>
                <audio id="audioPlayback" controls></audio>
            </div>
            
            <div id="result"></div>
        </div>

        <script>
            const form = document.getElementById('audioForm');
            const resultDiv = document.getElementById('result');
            const audioInput = document.getElementById('audio');
            const uploadedAudioPlayer = document.getElementById('uploadedAudioPlayback');

            // Handle form submission for uploaded audio files
            form.addEventListener('submit', async function(event) {
                event.preventDefault();
                
                const formData = new FormData(form);
                resultDiv.style.display = 'block';
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

            // Audio preview for uploaded files
            audioInput.addEventListener('change', function(event) {
                const file = event.target.files[0];
                if (file) {
                    const audioUrl = URL.createObjectURL(file);
                    uploadedAudioPlayer.src = audioUrl;
                    uploadedAudioPlayer.style.display = 'block';  // Show the player once a file is selected
                } else {
                    uploadedAudioPlayer.style.display = 'none';  // Hide the player if no file is selected
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
                        uploadRecordedButton.style.display = 'inline-block';

                        // Upload the recorded audio when ready
                        uploadRecordedButton.addEventListener('click', async () => {
                            const formData = new FormData();
                            formData.append('audio', audioBlob, 'recorded_audio.wav');
                            resultDiv.style.display = 'block';
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

                    startRecordButton.style.display = 'none';
                    stopRecordButton.style.display = 'inline-block';
                } catch (error) {
                    console.error('Error accessing microphone:', error);
                }
            });

            // Stop recording
            stopRecordButton.addEventListener('click', () => {
                mediaRecorder.stop();
                startRecordButton.style.display = 'inline-block';
                stopRecordButton.style.display = 'none';
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
