
# from flask import Flask, request, jsonify, send_file
# import numpy as np
# import joblib
# from PIL import Image
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
# import os
# from flask_cors import CORS
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import numpy  as np
# from flask_sqlalchemy import SQLAlchemy
# from flask_bcrypt import Bcrypt
# from flask_jwt_extended import JWTManager, create_access_token
# from sklearn.metrics import classification_report
# from sklearn.model_selection import cross_val_score
# from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
# from sqlalchemy import text


# app = Flask(__name__)
# CORS(app)


# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # Change to PostgreSQL or MySQL in production
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# app.config['JWT_SECRET_KEY'] = 'your_secret_key'  # Change this in production
# app.config['PROPAGATE_EXCEPTIONS'] = True

# db = SQLAlchemy(app)
# bcrypt = Bcrypt(app)
# jwt = JWTManager(app)

# # User model
# class User(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     full_name = db.Column(db.String(100), nullable=False)
#     email = db.Column(db.String(100), unique=True, nullable=False)
#     mobile_number = db.Column(db.String(15), unique=True, nullable=False)
#     password = db.Column(db.String(200), nullable=False)

# # with app.app_context():
# #     with db.engine.connect() as conn:
# #         conn.execute(text("ALTER TABLE user ADD COLUMN mobile_number VARCHAR(15);"))
# #         print("Column added successfully!")

# # Load the trained SVM model, PCA transformer, and CNN weights
# svm_model = joblib.load("svm_model.pth")
# pca_transformer = joblib.load("pca_transformer.pth")
# conf_matrix = np.load("conf_matrix.npy")  # Ensure you save the confusion matrix as .npy
# metrics = joblib.load("metrics.pkl") 
# classification_metrics = joblib.load("classification_report.pkl")
# cross_val_results = joblib.load("cross_validation.pkl")

# # Rebuild CNN feature extractor
# cnn_model = Sequential([
#     Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)),
#     MaxPooling2D((2, 2)),
#     BatchNormalization(),
#     Conv2D(32, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     BatchNormalization(),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     BatchNormalization(),
#     Flatten()
# ])
# cnn_model.load_weights("cnn_weights.h5")


# def save_confusion_matrix():
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
#     plt.xlabel("Predicted Label")
#     plt.ylabel("True Label")
#     plt.title("Confusion Matrix")
#     plt.savefig("static/metrics/confusion_matrix.jpg")
#     plt.close()
# save_confusion_matrix()

# # Classification labels
# classes = ['Non Demented', 'Very Mild Demented', 'Mild Demented', 'Moderate Demented']

# # Function to preprocess image
# def preprocess_image(image_path):
#     img = Image.open(image_path).convert('RGB')
#     img = img.resize((128, 128))
#     img_array = np.array(img)
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     return img_array

# @app.route("/predict", methods=["POST"])
# def predict():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
    
#     file = request.files["file"]
#     image_path = os.path.join("static", file.filename)
#     file.save(image_path) 
#     # Preprocess image
#     input_image = preprocess_image(image_path)
#     # Extract CNN features
#     image_features = cnn_model.predict(input_image)
#     # Apply PCA transformation
#     image_reduced = pca_transformer.transform(image_features)
#     # Predict using SVM
#     predicted_class = svm_model.predict(image_reduced)[0]
#     print(f"Predicted class: ",classes[predicted_class])
#     result = {
#         "prediction": classes[predicted_class]
#     }
#     return jsonify(result)


# @app.route("/metrics", methods=["GET"])
# def get_metrics():
#     return jsonify(metrics)

# @app.route("/confusion-matrix", methods=["GET"])
# def get_confusion_matrix():
#     return send_file("static/metrics/confusion_matrix.jpg", mimetype="image/png")

# def save_classification_report_image(report_data, filename="static/metrics/classification_report.jpg"):
#     df = pd.DataFrame(report_data).T
#     df["support"] = df["support"].astype(int)  # Ensure 'support' is integer

#     plt.figure(figsize=(8, 4))
#     sns.heatmap(df, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)
#     plt.title("Classification Report")
#     plt.savefig(filename, bbox_inches="tight")
#     plt.close()

# # Generate the classification report image
# save_classification_report_image(classification_metrics)

# @app.route("/classification-report", methods=["GET"])
# def get_classification_report():
#     return send_file("static/metrics/classification_report.jpg", mimetype="image/png")


# def load_cross_validation_results():
#     try:
#         cross_val_results = joblib.load("cross_validation.pkl")  # Update the path if needed

#         if isinstance(cross_val_results, np.ndarray):
#             cross_val_results = cross_val_results.tolist()  # Convert to list if numpy array

#         return cross_val_results if isinstance(cross_val_results, list) else []

#     except Exception as e:
#         print(f"Error loading cross-validation results: {e}")
#         return []

# @app.route("/cross-validation", methods=["GET"])
# def get_cross_validation():
#     cross_val_results = load_cross_validation_results()

#     # ✅ Check if list is valid (not empty and contains only valid numbers)
#     if isinstance(cross_val_results, list) and all(isinstance(x, (int, float)) for x in cross_val_results) and len(cross_val_results) > 0:
#         return jsonify({
#             "folds": list(range(1, len(cross_val_results) + 1)), 
#             "accuracies": cross_val_results
#         })
#     else:
#         return jsonify({"error": "Cross-validation results are empty or invalid"}), 400  # Return HTTP 400 Bad Request

# @app.route('/static/uploads/<filename>')
# def serve_image(filename):
#     return send_file(os.path.join("static/uploads", filename))


# # Create tables before first request
# with app.app_context():
#     db.create_all()

# # Signup route
# @app.route('/signup', methods=['POST'])
# def signup():
#     data = request.get_json()
#     full_name = data.get('fullName')
#     email = data.get('email')
#     mobile_number=data.get('mobileNumber')
#     password = data.get('password')
    
#     if User.query.filter_by(email=email).first():
#         return jsonify({'error': 'Email already registered'}), 400
#     if User.query.filter_by(mobile_number=mobile_number).first():
#         return jsonify({'error': 'Mobile number already registered'}), 400
    
#     hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
#     new_user = User(full_name=full_name, email=email, mobile_number=mobile_number, password=hashed_password)
#     db.session.add(new_user)
#     db.session.commit()
#     return jsonify({'message': 'User registered successfully'}), 201

# # Login route
# @app.route('/login', methods=['POST'])
# def login():
#     data = request.get_json()
#     email = data.get('email')
#     password = data.get('password')
    
#     user = User.query.filter_by(email=email).first()
#     if user and bcrypt.check_password_hash(user.password, password):
#         access_token = create_access_token(identity=user.email)
#         return jsonify({
#             "token": access_token,
#             "user": {
#                 "id": user.id,
#                 "full_name": user.full_name,
#                 "email": user.email,
#                 "mobile_number": user.mobile_number,
#                 "role": "User"
#             }
#         })
    
#     return jsonify({'error': 'Invalid email or password'}), 401

# @app.route('/delete-account', methods=['DELETE'])
# # @jwt_required()  # Ensures only authenticated users can delete their account
# def delete_account():
#     data = request.get_json()
#     user_id = data.get('id')  # Get the user ID from the request

#     if not user_id:
#         return jsonify({'error': 'User ID is required'}), 400

#     user = User.query.get(user_id)  # Find the user by ID

#     if not user:
#         return jsonify({'error': 'User not found'}), 404 

#     db.session.delete(user)  # Delete the user
#     db.session.commit()  # Commit the changes

#     return jsonify({'message': 'Account deleted successfully'}), 200

# #update route
# @app.route('/update-profile', methods=['PUT'])
# #  @jwt_required()  # Requires authentication
# def update_profile():
#     data = request.get_json()
#     user_id = data.get('id')
#     new_name = data.get('full_name')
#     new_mobile_number = data.get('mobile_number')


#     user = User.query.get(user_id)
#     if not user:
#         return jsonify({'error': 'User not found'}), 404

#     user.full_name = new_name
#     user.mobile_number = new_mobile_number

#     db.session.commit()
#     return jsonify({'message': 'Profile updated successfully'}), 200

# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, request, jsonify, send_file
import numpy as np
import joblib
from PIL import Image
import tensorflow as tf

import os
from flask_cors import CORS
import matplotlib.pyplot as plt
# from tensorflow.keras.models import Sequential
from keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
import seaborn as sns
import pandas as pd
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask import Flask, request, jsonify, send_from_directory



app = Flask(__name__)
CORS(app)
# CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}, supports_credentials=True, allow_headers=["Authorization", "Content-Type"], methods=["GET", "POST", "DELETE", "OPTIONS"])

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = 'your_secret_key'
app.config['PROPAGATE_EXCEPTIONS'] = True

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    mobile_number = db.Column(db.String(15), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    # profile_image = db.Column(db.String(255), nullable=True, default="static/uploads/default_profile.jpg")
# from sqlalchemy.inspection import inspect

# with app.app_context():
#     inspector = inspect(db.engine)
#     print(inspector.get_columns("user"))
# Create tables before first request
with app.app_context():
    db.create_all()

svm_model = joblib.load("svm_model.pth")
pca_transformer = joblib.load("pca_transformer.pth")
conf_matrix = np.load("conf_matrix.npy")  # Ensure you save the confusion matrix as .npy
metrics = joblib.load("metrics.pkl") 
classification_metrics = joblib.load("classification_report.pkl")
cross_val_results = joblib.load("cross_validation.pkl")

# Rebuild CNN feature extractor
cnn_model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    Flatten()
])
cnn_model.load_weights("cnn_weights.h5")


def save_confusion_matrix():
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig("static/metrics/confusion_matrix.jpg")
    plt.close()
save_confusion_matrix()

# Classification labels
classes = ['Non Demented', 'Very Mild Demented', 'Mild Demented', 'Moderate Demented']

# Function to preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    image_path = os.path.join("static", file.filename)
    file.save(image_path) 
    # Preprocess image
    input_image = preprocess_image(image_path)
    # Extract CNN features
    image_features = cnn_model.predict(input_image)
    # Apply PCA transformation
    image_reduced = pca_transformer.transform(image_features)
    # Predict using SVM
    predicted_class = svm_model.predict(image_reduced)[0]
    print(f"Predicted class: ",classes[predicted_class])
    result = {
        "prediction": classes[predicted_class]
    }
    return jsonify(result)


@app.route("/metrics", methods=["GET"])
def get_metrics():
    return jsonify(metrics)

@app.route("/confusion-matrix", methods=["GET"])
def get_confusion_matrix():
    return send_file("static/metrics/confusion_matrix.jpg", mimetype="image/png")

def save_classification_report_image(report_data, filename="static/metrics/classification_report.jpg"):
    df = pd.DataFrame(report_data).T
    df["support"] = df["support"].astype(int)  # Ensure 'support' is integer

    plt.figure(figsize=(8, 4))
    sns.heatmap(df, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)
    plt.title("Classification Report")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

# Generate the classification report image
save_classification_report_image(classification_metrics)

@app.route("/classification-report", methods=["GET"])
def get_classification_report():
    return send_file("static/metrics/classification_report.jpg", mimetype="image/png")


def load_cross_validation_results():
    try:
        cross_val_results = joblib.load("cross_validation.pkl")  # Update the path if needed

        if isinstance(cross_val_results, np.ndarray):
            cross_val_results = cross_val_results.tolist()  # Convert to list if numpy array

        return cross_val_results if isinstance(cross_val_results, list) else []

    except Exception as e:
        print(f"Error loading cross-validation results: {e}")
        return []

@app.route("/cross-validation", methods=["GET"])
def get_cross_validation():
    cross_val_results = load_cross_validation_results()

    # ✅ Check if list is valid (not empty and contains only valid numbers)
    if isinstance(cross_val_results, list) and all(isinstance(x, (int, float)) for x in cross_val_results) and len(cross_val_results) > 0:
        return jsonify({
            "folds": list(range(1, len(cross_val_results) + 1)), 
            "accuracies": cross_val_results
        })
    else:
        return jsonify({"error": "Cross-validation results are empty or invalid"}), 400  # Return HTTP 400 Bad Request

@app.route('/static/uploads/<filename>')
def serve_image(filename):
    return send_file(os.path.join("static/uploads", filename))



# Signup route
@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    full_name = data.get('fullName')
    email = data.get('email')
    mobile_number = data.get('mobileNumber')
    password = data.get('password')
    
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already registered'}), 400
    if User.query.filter_by(mobile_number=mobile_number).first():
        return jsonify({'error': 'Mobile number already registered'}), 400
    
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(full_name=full_name, email=email, mobile_number=mobile_number, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'message': 'User registered successfully'}), 201

# Login route
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')      
    password = data.get('password')
    
    user = User.query.filter_by(email=email).first()
    if user and bcrypt.check_password_hash(user.password, password):
        # access_token = create_access_token(identity={'email': user.email, 'name': user.full_name, 'mobile_number': user.mobile_number})
        access_token = create_access_token(identity=user.email)
        return jsonify({
            "token": access_token,
            "user": {
                "id": user.id,
                "full_name": user.full_name,
                "email": user.email,
                "mobile_number": user.mobile_number,
                "role": "User"
            }
        })  
    return jsonify({'error': 'Invalid email or password'}), 401

#delete route
@app.route('/delete-account', methods=['DELETE'])
# @jwt_required()  # Ensures only authenticated users can delete their account
def delete_account(): 
    data = request.get_json()
    user_id = data.get('id')  # Get the user ID from the request

    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400

    user = User.query.get(user_id)  # Find the user by ID

    if not user:
        return jsonify({'error': 'User not found'}), 404

    db.session.delete(user)  # Delete the user
    db.session.commit()  # Commit the changes

    return jsonify({'message': 'Account deleted successfully'}), 200

#update route
@app.route('/update-profile', methods=['PUT'])
#  @jwt_required()  # Requires authentication
def update_profile():
    data = request.get_json()
    user_id = data.get('id')
    new_name = data.get('full_name')
    new_mobile_number = data.get('mobile_number')
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404

    user.full_name = new_name
    user.mobile_number = new_mobile_number

    db.session.commit()
    return jsonify({'message': 'Profile updated successfully'}), 200

if __name__ == "__main__":
    app.run(debug=True)