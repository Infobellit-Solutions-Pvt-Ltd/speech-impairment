import json
import os
from flask import request, jsonify

# Define the path to the user storage file (JSON format)
USER_STORAGE_FILE = "user_details.json"

def load_user_data():
    """Load existing user details from the storage file."""
    if os.path.exists(USER_STORAGE_FILE):
        with open(USER_STORAGE_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                # Handle the case where the file is empty or contains invalid JSON
                return {}
    return {}


def save_user_data(data):
    """Save updated user details to the storage file."""
    with open(USER_STORAGE_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def store_user():
    """Store Google login details and create a user ID."""
    user_data = request.json

    # Ensure the 'sub' field exists in the provided data
    #if 'uid' not in user_data:
        #return jsonify({"error": "Missing 'uid' field in user data"}), 400
    if 'uid' not in user_data and 'sub' not in user_data:
    	return jsonify({"error": "Missing both 'uid' and 'sub' fields in user data"}), 400

    
    # Generate a unique user_id
    user_id = f"user-{user_data['uid']}"

    # Load existing user data
    stored_data = load_user_data() 
    
    # Update or add new user details
    stored_data[user_id] ={
    "providerId": user_data.get("providerId"),
    "uid": user_data.get("uid"),
    "displayName": user_data.get("displayName"),
    "email": user_data.get("email"),
    "phoneNumber": user_data.get("phoneNumber"),
    "photoURL": user_data.get("photoURL"),
    #"user_id": user_data.get("user_id")
    "user_id": user_data.get("user_id") if 'user_id' in user_data else user_id  # Set user_id to a generated ID if missing
    }
    
    
    # Save the updated data to the storage file
    save_user_data(stored_data)

    return jsonify({
        "message": "User details stored successfully",
        #"user_id": user_id,
        "stored_data": stored_data[user_id]
    }), 200
#====================================================================================
