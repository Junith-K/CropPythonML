from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from io import BytesIO
import joblib
import pandas as pd

app = Flask(__name__)

model_path1 = '1/'
model1 = tf.keras.models.load_model(model_path1)
model_path2 = '2/'
model2 = tf.keras.models.load_model(model_path2)
model_path31 = '3/1/areaproduction.sav'
model31 = joblib.load(model_path31)
model_path32 = '3/2/rainfall_model.sav'
model32 = joblib.load(model_path32)

class_names1 = ['crop','weed'] 
class_names2=['final_coatbuttons_jute1', 'final_common_poppy_wheat1', 'final_corn1', 'final_dayflower_sugarcane1', 'final_foxtail_maize1', 'final_rice1', 'final_sugarcane1', 'final_wheat1']
data = pd.read_csv('3/area_production_data.csv')
data.fillna(data.mean(numeric_only=True), inplace=True)
data = pd.get_dummies(data)

X = data.drop(['Area', 'Production'], axis=1)

def predict1(img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model1.predict(img_array)

    predicted_class = class_names1[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence

@app.route('/predict1', methods=['POST'])
def predict_image1():
    try:
        file = request.files['image']
        img = tf.keras.preprocessing.image.load_img(BytesIO(file.read()), target_size=(512, 512))
        predicted_class, confidence = predict1(img)
        return jsonify({'predicted_class': predicted_class, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)})
    

    
def predict2(img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model2.predict(img_array)

    predicted_class = class_names2[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence

@app.route('/predict2', methods=['POST'])
def predict_image2():
    try:
        file = request.files['image']
        img = tf.keras.preprocessing.image.load_img(BytesIO(file.read()), target_size=(390, 260))
        predicted_class, confidence = predict2(img)
        return jsonify({'predicted_class': predicted_class, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)})
    

def preprocess_input(input_data):
    input_df = pd.DataFrame(input_data, index=[0])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    return input_df

@app.route('/predict31', methods=['POST'])
def predict_image31():
    try:
        input_data = request.get_json()  # Get data from the JSON body
        input_df = preprocess_input(input_data)

        output = model31.predict(input_df)
        print(output)
        output_list = output.tolist()  # Convert ndarray to Python list
        return jsonify({'area': output_list[0][0], 'production': output_list[0][1]}) 
    except Exception as e:
        return jsonify({'error': str(e)})
    

@app.route('/predict32', methods=['POST'])
def predict_image32():
    try:
        input_data = request.get_json()  # Get data from the JSON body
        manual_data = pd.DataFrame(input_data)
        manual_prediction = model32.predict(manual_data)
        print('Manual Prediction:')
        print(manual_prediction)
        output_list = manual_prediction.tolist()
        return jsonify({'JFMA': output_list[0][0], 'MJJA': output_list[0][1], 'SOND': output_list[0][2]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)


# input_data31 = {'State_name': 'Andhra Pradesh', 'District_Name': 'ANANTAPUR', 'Crop_Year': 1999, 'Season': 'Kharif', 'Crop': 'Bajra'}
# input_data32 = {'SUBDIVISION': [1], 'YEAR': [2015]}