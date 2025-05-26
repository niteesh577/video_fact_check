import os
import json
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import tempfile
import shutil
from main import process_video
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For flash messages

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'mkv'}

# Create upload and results directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file was uploaded
    if 'video' not in request.files:
        flash('No file part', 'danger')
        return redirect(request.url)
    
    file = request.files['video']
    
    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Process the video
        try:
            logger.info(f"Processing video: {file_path}")
            result = process_video(file_path)
            
            # Save the result to a JSON file
            result_filename = f"{os.path.splitext(filename)[0]}_result.json"
            result_path = os.path.join(RESULTS_FOLDER, result_filename)
            
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Redirect to the results page
            return redirect(url_for('show_result', filename=filename))
        
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            flash(f"Error processing video: {str(e)}", 'danger')
            return redirect(request.url)
    
    flash('Invalid file type. Allowed types: mp4, avi, mov, wmv, mkv', 'danger')
    return redirect(request.url)

@app.route('/result/<filename>')
def show_result(filename):
    # Get the result file path
    result_filename = f"{os.path.splitext(filename)[0]}_result.json"
    result_path = os.path.join(RESULTS_FOLDER, result_filename)
    
    # Check if the result file exists
    if not os.path.exists(result_path):
        flash('Result file not found', 'danger')
        return redirect(url_for('index'))
    
    # Load the result
    with open(result_path, 'r') as f:
        result = json.load(f)
    
    # Render the result template
    return render_template('result.html', 
                           filename=filename, 
                           verdict=result.get('verdict', 'Unknown'),
                           analysis=result.get('analysis', 'No analysis available'),
                           summary=result.get('summary', 'No summary available'),
                           result_json=json.dumps(result))

@app.route('/api/detect', methods=['POST'])
def api_detect():
    # Check if a file was uploaded
    if 'video' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['video']
    
    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name
        
        try:
            # Process the video
            logger.info(f"Processing video via API: {file.filename}")
            result = process_video(temp_file_path)
            
            # Clean up the temporary file
            os.unlink(temp_file_path)
            
            # Return the result
            return jsonify(result)
        
        except Exception as e:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            
            logger.error(f"API error: {str(e)}")
            return jsonify({'error': f"Error processing video: {str(e)}"}), 500
    
    return jsonify({'error': 'Invalid file type. Allowed types: mp4, avi, mov, wmv, mkv'}), 400

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the Flask web application for deepfake detection')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    app.run(debug=args.debug, host=args.host, port=args.port)