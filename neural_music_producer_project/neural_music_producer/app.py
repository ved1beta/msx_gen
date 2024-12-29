from flask import Flask, render_template, request, send_file, jsonify
import os
from werkzeug.utils import secure_filename
from main import main
import subprocess

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_midi_to_wav(midi_file):
    wav_file = midi_file.replace('.mid', '.wav')
    try:
        subprocess.run(['timidity', midi_file, '-Ow', '-o', wav_file], check=True)
        return wav_file
    except subprocess.CalledProcessError as e:
        print(f"Error converting MIDI to WAV: {e}")
        raise Exception("Failed to convert MIDI to WAV. Please ensure TiMidity is installed.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/convert', methods=['POST'])
def convert_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process the video using your existing pipeline
            main(filepath)
            
            # Convert the generated MIDI to WAV
            midi_file = "output_music.mid"
            wav_file = convert_midi_to_wav(midi_file)
            
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

@app.route('/download')
def download_audio():
    wav_file = 'output_music.wav'
    if not os.path.exists(wav_file):
        return jsonify({'error': 'Audio file not found'}), 404
    
    return send_file(wav_file,
                    mimetype='audio/wav',
                    as_attachment=True,
                    download_name='generated_music.wav')

if __name__ == '__main__':
    app.run(debug=True)