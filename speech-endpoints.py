import os
import random
import json
import tempfile
import whisper
import time
import psutil
import boto3
from flask_cors import CORS
from flask import Flask, request, jsonify, send_file
from flask_socketio import SocketIO, emit
from TTS.api import TTS
from pydub import AudioSegment
from werkzeug.utils import secure_filename
from utils.user_storage import store_user, load_user_data
from botocore.exceptions import NoCredentialsError, ClientError
from dotenv import load_dotenv
#=============================================================================================
app = Flask(__name__, static_folder="./build", static_url_path="/")
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#=============================================================================================
# Load environment variables from .env file
load_dotenv()

# Access the environment variables
AWS_REGION = os.getenv('AWS_REGION')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
S3_BUCKET = os.getenv('S3_BUCKET')

# Configure AWS S3
s3_client = boto3.client(
    's3',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# Now `s3_client` can interact with your S3 bucket
print(f"Connected to S3 bucket: {S3_BUCKET}")

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
#======================================================================================================
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
#============================================================================================
def upload_to_s3(file_path, filename, bucket_name):
    """Helper function to upload files to S3."""
    s3_key = f"audio/{filename}"
    try:
        s3_client.upload_file(file_path, bucket_name, s3_key)
        s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
        print(f"File uploaded successfully: {s3_url}")
        return s3_url
    except FileNotFoundError:
        print(f"The file was not found: {file_path}")
        return None
    except NoCredentialsError:
        print("Credentials not available")
        return None
    except Exception as e:
        print(f"Error uploading to S3: {str(e)}")
        return None

#=============================================================================================


def delete_from_s3(bucket_name, filenames):
    """Helper function to delete multiple files from S3 using only file names."""
    s3_client = boto3.client('s3')

    # Construct the S3 keys from filenames
    objects_to_delete = [{'Key': f'audio/{filename}'} for filename in filenames]

    try:
        response = s3_client.delete_objects(
            Bucket=bucket_name,
            Delete={
                'Objects': objects_to_delete
            }
        )
        
        # Check if any files were successfully deleted
        deleted_files = response.get('Deleted', [])
        if deleted_files:
            print(f"Deleted files: {[obj['Key'] for obj in deleted_files]}")
        else:
            print("No files were deleted.")
        
        # Check for errors
        if 'Errors' in response:
            for error in response['Errors']:
                print(f"Error deleting {error['Key']}: {error['Message']}")
    except NoCredentialsError:
        print("Credentials not available")
    except ClientError as e:
        print(f"Error deleting from S3: {str(e)}")

#============================================================================================
# Function to rename an S3 file
def rename_s3_file(bucket_name, old_filename, new_filename):
    """Helper function to rename a file in S3 by copying it and then deleting the old one."""
    try:
        copy_source = {'Bucket': bucket_name, 'Key': f"audio/{old_filename}"}
        s3_client.copy_object(Bucket=bucket_name, CopySource=copy_source, Key=f"audio/{new_filename}")
        s3_client.delete_object(Bucket=bucket_name, Key=f"audio/{old_filename}")
        new_s3_url = f"https://{bucket_name}.s3.amazonaws.com/audio/{new_filename}"
        return new_s3_url
    except Exception as e:
        print(f"Error renaming file: {str(e)}")
        return None
#============================================================================================
def save_metadata(metadata, filename="metadata.json"):
    """Helper function to save metadata to a single JSON file on the server.
    If the file exists, append the new metadata; otherwise, create the file."""
    
    # Define the path to save the JSON file in the server's working directory
    metadata_filepath = os.path.join(os.getcwd(), filename)
    
    # Try to load existing metadata from the file, if it exists
    if os.path.exists(metadata_filepath):
        try:
            with open(metadata_filepath, 'r') as json_file:
                existing_data = json.load(json_file)
        except json.JSONDecodeError:
            # If file exists but is empty or invalid, start with an empty list
            existing_data = []
    else:
        # If the file doesn't exist, start with an empty list
        existing_data = []
    
    # Append the new metadata to the existing data
    existing_data.append(metadata)
    
    # Write the updated data back to the file
    try:
        with open(metadata_filepath, 'w') as json_file:
            json.dump(existing_data, json_file, indent=4)
        print(f"Metadata appended and saved at {metadata_filepath}")
        return metadata_filepath
    except Exception as e:
        print(f"Error saving metadata: {str(e)}")
        return None
#===========================================================================================


# Initialize S3 client
s3_client = boto3.client('s3')

def create_presigned_url(bucket_name, object_name, expiration=3600):
    """
    Generate a presigned URL to share an S3 object.
    :param bucket_name: string
    :param object_name: string
    :param expiration: Time in seconds for the presigned URL to remain valid (default is 1 hour)
    :return: Presigned URL as string. If error, returns None.
    """
    try:
        response = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': bucket_name, 'Key': object_name},
                                                    ExpiresIn=expiration)
    except Exception as e:
        print(f"Error generating presigned URL: {e}")
        return None
    return response
#=============================================================================================

@app.route('/')
def hello_world():
    return "<h1>Hey there! <br> Backend script running here!!</h1>"
#=======================================================================


@app.route('/save_metadata', methods=['POST'])
def save_metadata_endpoint():
    """Endpoint to save arbitrary metadata to a JSON file."""
    metadata = request.get_json()

    if not metadata:
        return jsonify({"error": "No metadata provided"}), 400

    # Generate a random number and get the current epoch time
    random_number = random.randint(1000, 9999)  # Adjust the range as needed
    current_epoch_time = int(time.time())
    
    # Create the id as randomnumber+current epochtime
    metadata['id'] = f"{random_number}_{current_epoch_time}"
    
    # print(metadata)
    data =  save_metadata(metadata)
    # print(data)
    return jsonify({"status": "Success"}), 200
#=============================================================================================
@app.route('/remove_audio_s3', methods=['POST'])
def delete_from_s3():
    """Helper function to delete multiple files from S3 using only file names."""
    s3_client = boto3.client('s3')

    # Construct the S3 keys from filenames

    inputJSON = request.get_json()

    objects_to_delete = [{'Key': f'audio/{filename}'} for filename in inputJSON['files']]

    try:
        response = s3_client.delete_objects(
            Bucket=S3_BUCKET,
            Delete={
                'Objects': objects_to_delete
            }
        )
        
        # Check if any files were successfully deleted
        deleted_files = response.get('Deleted', [])
        if deleted_files:
            print(f"Deleted files: {[obj['Key'] for obj in deleted_files]}")
            return jsonify({"status": "Success"}), 200

        else:
            print("No files were deleted.")
            return jsonify({"status": "failed"}), 200
        # Check for errors
        if 'Errors' in response:
            for error in response['Errors']:
                print(f"Error deleting {error['Key']}: {error['Message']}")
    except NoCredentialsError:
        print("Credentials not available")
    except ClientError as e:
        print(f"Error deleting from S3: {str(e)}")
#=============================================================================================
@app.route('/remove_record', methods=['POST'])
def remove_record_by_id(filename="metadata.json"):
    """
    Remove a specific record from the JSON file by matching the provided id.
    """
    inputJSON = request.get_json()

    # Define the path to the JSON file
    metadata_filepath = os.path.join(os.getcwd(), filename)

    # Check if the file exists
    if not os.path.exists(metadata_filepath):
        print(f"No file found at {metadata_filepath}")
        return jsonify({"status": "failed", "message": "Metadata file not found"}), 404

    # Load existing metadata
    try:
        with open(metadata_filepath, 'r') as json_file:
            data = json.load(json_file)
    except json.JSONDecodeError:
        print("File is empty or corrupted.")
        return jsonify({"status": "failed", "message": "Metadata file corrupted"}), 400

    # Filter out the record with the matching id
    updated_data = [record for record in data if record.get("id") != inputJSON.get('id')]

    # Write the updated data back to the file
    try:
        with open(metadata_filepath, 'w') as json_file:
            json.dump(updated_data, json_file, indent=4)

        # Collect files to delete if specified
        filesList = [inputJSON.get('inputFile'), inputJSON.get('outputFile')]
        filesList = [file for file in filesList if file]  # Filter out None values

        if filesList:
            objects_to_delete = [{'Key': f'audio/{filename}'} for filename in filesList]
            response = s3_client.delete_objects(
                Bucket=S3_BUCKET,
                Delete={'Objects': objects_to_delete}
            )
            
            # Check if any files were successfully deleted
            deleted_files = response.get('Deleted', [])
            if deleted_files:
                print(f"Deleted files: {[obj['Key'] for obj in deleted_files]}")
            else:
                print("No files were deleted.")

        print(f"Record with id '{inputJSON['id']}' removed from {metadata_filepath}")
        return jsonify({"status": "Success", "message": f"Record with id '{inputJSON['id']}' removed"}), 200

    except Exception as e:
        print(f"Error updating metadata: {str(e)}")
        return jsonify({"status": "failed", "message": str(e)}), 500
#=============================================================================================
@app.route('/process_audio', methods=['POST'])
def process_audio():
    """Combined endpoint for transcribing and generating speech using the same uploaded audio for cloning."""
    
    # Check if the required fields are present
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    # Retrieve the audio file and other parameters from the request
    audio_file = request.files['audio']
    audio_filename = secure_filename(audio_file.filename)

    # Extract user-provided parameters
    user_id = request.form.get('user_id', 'NO_ID')  # Use 'NO_ID' if not provided
    input_type = request.form.get('type', 'speech')  # Default type is 'speech'
    
    current_epoch_time = int(time.time())
    
    # Get additional metadata fields with defaults
    transcription_text = request.form.get('input', 'Default transcription')
    input_filename = request.form.get('inputFile', audio_filename)
    duration = request.form.get('duration', 'Unknown duration')
    file_type = request.form.get('fileType', audio_file.content_type)
    date_and_time = request.form.get('dateAndtime', time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(current_epoch_time)))

    try:
        # Measure total response time
        response_start_time = time.time()

        # Save the uploaded audio file temporarily
        input_filename = f"{user_id}_input_{current_epoch_time}.wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
            tmp_audio_file.write(audio_file.read())
            tmp_audio_path = tmp_audio_file.name

        # Upload input audio file to S3 with proper naming
        input_audio_s3_url = upload_to_s3(tmp_audio_path, input_filename, S3_BUCKET)

        # Measure transcription time (Whisper)
        transcription_start_time = time.time()
        result = whisper_model.transcribe(tmp_audio_path, language='en')
        transcription_text = result['text'].strip()
        transcription_time = time.time() - transcription_start_time
        print(f"Transcription completed in {transcription_time:.2f} seconds.")

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
                    speaker_wav=tmp_audio_path,
                    language="en"
                )
                
                # Load the generated chunk audio and append it to the final audio
                chunk_audio = AudioSegment.from_wav(tmp_output_file.name)
                final_audio += chunk_audio

        tts_generation_time = time.time() - tts_start_time
        print(f"TTS generation completed in {tts_generation_time:.2f} seconds.")

        # Save the final concatenated audio to a new file
        output_filename = f"{user_id}_output_{current_epoch_time}.wav"
        combined_output_path = os.path.join(tempfile.gettempdir(), output_filename)
        final_audio.export(combined_output_path, format="wav")

        # Upload output audio file to S3
        output_audio_s3_url = upload_to_s3(combined_output_path, output_filename, S3_BUCKET)

        # Clean up temporary files
        os.remove(tmp_audio_path)

        # Measure total response time
        total_response_time = time.time() - response_start_time
        print(f"Total time taken to generate response: {total_response_time:.2f} seconds.")

        

        return jsonify({
            "message": "Process completed and metadata saved",
            # "uid": metadata_id,
            "user_id": user_id,
            "transcription": transcription_text,
            "input_audio_url": input_filename,  # Correctly reference input audio URL
            "generated_speech_url": output_filename
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
#=============================================================================================

@app.route('/update_filename', methods=['POST'])
def update_filename():
    """Endpoint to rename an existing file in S3."""
    data = request.json
    old_filename = data.get('old_filename')
    new_filename = data.get('new_filename')

    if not old_filename or not new_filename:
        return jsonify({"error": "Both old and new file names are required."}), 400

    # Rename the file in S3
    new_s3_url = rename_s3_file(S3_BUCKET, old_filename, new_filename)
    if new_s3_url:
        return jsonify({
            "message": "File renamed successfully",
            "new_file_url": new_s3_url
        })
    else:
        return jsonify({"error": "File renaming failed."}), 500
        
#==============================================================================================
@app.route('/store_user', methods=['POST'])
def handle_store_user():
    """
    Endpoint to store user details from Google login.

    Expects a JSON payload with user information including 'sub', 'email', 
    'email_verified', 'name', 'given_name', 'family_name', and 'picture'.
    """

    # Call the utility function to store user data
    response = store_user()  

    # Return the response from the utility function
    return response  

#==============================================================================================
@app.route('/temp_url', methods=['GET'])
def get_file_url():
    file_name = request.args.get('fileName')

    if not file_name:
        return jsonify({"error": "Missing 'fileName' parameter"}), 400

    try:
        # Check if the file exists in S3 with proper key
        s3_client.head_object(Bucket=S3_BUCKET, Key=f"audio/{file_name}")
        
        # Generate presigned URL using the correct S3 client
        signed_url = create_presigned_url(S3_BUCKET, f"audio/{file_name}")

        if signed_url:
            return jsonify({"temp_URL": signed_url})
        else:
            return jsonify({"error": f"Failed to generate signed URL for '{file_name}'"}), 500

    except NoCredentialsError:
        return jsonify({"error": "AWS credentials are not available."}), 500
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return jsonify({"error": f"File '{file_name}' not found in S3 bucket."}), 404
        else:
            return jsonify({"error": "Error checking file in S3."}), 500

#==============================================================================================
@app.route('/get_metadata/<user_id>', methods=['GET'])
def get_user(user_id):
    """Retrieve user details based on user_id."""
    stored_data = load_user_data()

    # Check if the user_id exists
    if user_id in stored_data:
        return jsonify({
            "user_id": user_id,
            "details": stored_data[user_id]
        }), 200
    else:
        return jsonify({"error": "User not found"}), 404
        
#==============================================================================================
@app.route('/get_user_records/<user_id>', methods=['GET'])
def get_user_records(user_id):
    """Endpoint to retrieve metadata records for a specific user based on user_id."""
    metadata_filename = "metadata.json"
    metadata_filepath = os.path.join(os.getcwd(), metadata_filename)

    # Check if the metadata file exists
    if not os.path.exists(metadata_filepath):
        return jsonify({"user_records": []}), 200  # Return empty list if no metadata found

    try:
        # Load the metadata from the file
        with open(metadata_filepath, 'r') as json_file:
            records_metadata = json.load(json_file)

        # Filter records for the given user_id
        user_records = [record for record in records_metadata if record.get("user_id") == user_id]

        if user_records:
            return jsonify({"user_records": user_records}), 200
        else:
            return jsonify({"user_records": [], "message": "No records found for the specified user"}), 200

    except json.JSONDecodeError:
        return jsonify({"error": "Error decoding metadata file."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500    
#==============================================================================================
@app.route('/get_records_metadata', methods=['GET'])
def get_records_metadata():
    """Endpoint to retrieve metadata of all uploaded records."""
    metadata_filename = "metadata.json"
    metadata_filepath = os.path.join(os.getcwd(), metadata_filename)

    # Check if the metadata file exists
    if not os.path.exists(metadata_filepath):
        return jsonify({"records_metadata": []}), 200  # Return empty list if no metadata found

    try:
        # Load the metadata from the file
        with open(metadata_filepath, 'r') as json_file:
            records_metadata = json.load(json_file)
            return jsonify({"records_metadata": records_metadata}), 200
    except json.JSONDecodeError:
        return jsonify({"error": "Error decoding metadata file."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
#==============================================================================================
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5050)


