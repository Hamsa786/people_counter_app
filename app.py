# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import torch
import os
from PIL import Image, ImageDraw  # Add ImageDraw here
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

db = SQLAlchemy(app)

# User model for database
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def count_people(image_path):
    # Load a larger YOLOv5 model for better accuracy
    global model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l')  # Changed from yolov5s to yolov5l
    
    # Load and process image
    img = Image.open(image_path)
    
    # Increase inference size significantly for better detection
    results = model(img, size=1280)  # Increased from 640 to 1280
    
    # Lower confidence threshold to detect more people
    confidence_threshold = 0.25  # Lowered from 0.3
    iou_threshold = 0.4  # Lowered for less aggressive NMS
    
    # Get predictions and filter for person class
    predictions = results.pred[0]
    people_detections = predictions[predictions[:, -1] == 0]
    
    # Filter by confidence
    confident_detections = people_detections[people_detections[:, 4] >= confidence_threshold]
    
    # Apply Non-Maximum Suppression
    nms_indices = torch.ops.torchvision.nms(
        confident_detections[:, :4],
        confident_detections[:, 4],
        iou_threshold
    )
    
    final_detections = confident_detections[nms_indices]
    
    return len(final_detections), final_detections

@app.route('/')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('signup'))
        
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Successfully registered! Please login.')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            session['username'] = username
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('home'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('home'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Count people and get detections
        people_count, detections = count_people(filepath)
        
        # Draw bounding boxes on the image
        img = Image.open(filepath)
        img_draw = img.copy()
        draw = ImageDraw.Draw(img_draw)
        
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            conf = det[4]
            
            # Draw rectangle
            draw.rectangle([(x1, y1), (x2, y2)], outline='red', width=2)
            # Add confidence score
            draw.text((x1, y1-10), f'{conf:.2f}', fill='red')
        
        # Save annotated image
        output_filename = f'annotated_{filename}'
        output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        img_draw.save(output_filepath)
        
        return render_template('result.html', 
                             filename=output_filename,
                             original_filename=filename,
                             people_count=people_count)
    
    flash('Invalid file type')
    return redirect(url_for('home'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)