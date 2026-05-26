from flask import Flask, redirect, url_for, request, render_template, session
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)


model_path = 'wound_vgg19five.h5'
model = load_model(model_path)
# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'upload/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def main():
    return render_template('main.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    return render_template('login.html')

@app.route("/logout")
def logout():
    return render_template('main.html')

@app.route('/performance')
def performance():
    return render_template('performance.html')

@app.route("/index")
def index():
    return render_template('index.html')

def predictions(img_path, model):
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    pred = np.argmax(model.predict(x)[0], axis=-1)

    if pred == 0:
        preds = 'Accident_images'
    elif pred == 1:
        preds = "Burns"
    elif pred == 2:
        preds = "Cut"
    elif pred == 3:
        preds = "Diabetic_foot_ulcers"
    elif pred == 4:
        preds = "Ingrown_nails"
    
    return preds
    

@app.route("/predicted", methods=['POST'])
def predicted():
    if 'imagefile' not in request.files:
        return "No file part"
    
    uploaded_file = request.files['imagefile']
    if uploaded_file.filename == '':
        return "No selected file"

    if uploaded_file and allowed_file(uploaded_file.filename):
        filename = secure_filename(uploaded_file.filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(img_path)
        
        # Get the prediction for the uploaded image
        prediction = predictions(img_path, model)
        
        # Pass the image file path and prediction result to the template
        img = os.path.join('static/', filename)
        cv2.imwrite(img, cv2.imread(img_path))
        
        return render_template('result2.html', prediction=prediction, img=img)
    
    return "Invalid file"

if __name__ == '__main__':
    app.run(debug=True)