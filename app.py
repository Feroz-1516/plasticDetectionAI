from flask import Flask, request, render_template, url_for, jsonify, send_from_directory,session,redirect
import requests
import os
import cv2
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
import torchvision
import torch
from ultralytics import YOLO
import piexif
import random
from functools import wraps
import pymongo
from passlib.hash import pbkdf2_sha256
import uuid
from datetime import datetime, timedelta
from flask_cors import CORS
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from waitress import serve
from bson.objectid import ObjectId
import imageio
from single_class import ensemble_singleClass
from multi_class import predict_rcnn_multi
import concurrent.futures
from alerts import send_alert
from dotenv import load_dotenv


# Load environment variables from the .env file
load_dotenv()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


app = Flask(__name__)
CORS(app)
CORS(app, resources={r"/api/*": {"origins": "https://aqua-vision-reports.streamlit.app"}})

app.secret_key =  b'N\xfb2M\x8b\x1f\x9cH\xde{5T\xe1\xee\x13\xfb'


camera = None
captured_image = None

# Set the upload folder for the media (images and videos)
UPLOAD_FOLDER_1 = os.path.join('upload', 'selected_files')
app.config['UPLOAD_FOLDER_1'] = UPLOAD_FOLDER_1


UPLOAD_FOLDER_2 = os.path.join('upload', 'taken_files')
app.config['UPLOAD_FOLDER_2'] = UPLOAD_FOLDER_2

# Configure static folder to serve CSS and other static files
app.static_folder = 'static'



#---------------------------------------------DATABASE CREDENTIALS---------------------------------------------------------
# Replace these values with your MongoDB Atlas credentials and cluster details


# Construct the MongoDB Atlas connection string
connection_string = "mongodb+srv://aqua:12345@cluster0.n99hlyd.mongodb.net/?retryWrites=true&w=majority"

# Create a MongoClient instance using the connection string
client = pymongo.MongoClient(connection_string)

# Access the desired database
db = client["connect"]
authority_collection = db["authority"]

# Now you can interact with the MongoDB Atlas database as before

#--------------------------------------------------------------------------------------------------------------------------
# Load a pretrained YOLOv8n model
# model = YOLO('models/yolo_v8.onnx',task="detect")

model = YOLO("models/single_class_models/YoloV8.onnx",task = "detect")

def inference_video(file_name,media_path):

    # Define path to the video file
    # Define the path to the output video file
    video_file_name = 'predicted_' + file_name
    video_file_path = os.path.join(app.config['UPLOAD_FOLDER_1'], video_file_name)


    # Define the confidence threshold for object detection
    confidence_threshold = 0.1
    iou= 0.2
    # Define a dictionary to map class IDs to labels
    class_labels = {0: "Plastic"}

    # Open the video file
    cap = cv2.VideoCapture(media_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    output_writer = imageio.get_writer(video_file_path, fps=30, codec='vp9')


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection on the frame
        results = model(frame, conf=confidence_threshold,iou=iou)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = map(int, box)
                class_id = int(label)
                confidence = round(score, 2)

                # Filter out low-confidence detections
                if confidence > confidence_threshold:
                    # Get the corresponding label from the dictionary
                    class_label = class_labels.get(class_id, "Unknown")

                    # Draw bounding box and label on the frame
                    color = (0, 255, 0)  # Green color
                    label_text = f"{class_label}: {confidence:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write the frame with bounding boxes to the output video
        output_writer.append_data(frame)


    # Release video capture and writer
    cap.release()
    output_writer.close()

    cv2.destroyAllWindows()

    return video_file_path


def moving_stat(image_path):
    
        # Load a model
    model = YOLO('models/moving_stat/mov_stat_best.onnx')  # pretrained YOLOv8n model

    # Run batched inference on a list of images
    results = model(image_path, task='detect')  # return a list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes.xyxy.numpy()
        probs = result.boxes.conf.numpy()
        lab = result.names
        cls = result.boxes.cls.numpy()
        mapped_labels = [lab[label] for label in cls]

    return boxes,probs,mapped_labels




import openai
def generate_explanations(bbox,image_width="3992pixels",image_height="2992pixels"):
    openai.api_key =  os.getenv('OPENAI_API_KEY')

    detected_boxes=[list(arr) for arr in bbox]

    num_plastics = len(detected_boxes)
    
    # Calculate the distribution of plastics across the image
    top_count = bottom_count = left_count = right_count = center_count = 0
    for box in detected_boxes:
        x_min, y_min, x_max, y_max = box
        if y_min < 0.5:  # Top half of the image
            top_count += 1
        else:  # Bottom half of the image
            bottom_count += 1
        
        if x_min < 0.5:  # Left half of the image
            left_count += 1
        else:  # Right half of the image
            right_count += 1
        
        if 0.25 <= x_min <= 0.75 and 0.25 <= y_min <= 0.75:  # Center of the image
            center_count += 1
    explanation = ""
    if center_count > num_plastics * 0.3:
        explanation+= f" The concentration of plastics is higher in the center of the image."
    elif center_count == 0:
        explanation+= f" The concentration of plastics is scattered across the image."
    else:
        explanation+= f" The concentration of plastics is higher in various parts of the image."
    # Add the selected advice statement
   


    messages = [
        {
            "role": "system",
            "content": "You are an Explainable AI which describes about the plastics in an image. Your job is to just give the details of the plastics required by the user in a precise way without asking additional information."
        }
    ]

    message = f"bounding_boxes = {bbox}  image_width = {image_width}  image_height = {image_height} , provide plastic details: count, plastic density percentage in the image, and brief about {explanation}. Response should be concise, structured. Dont give any formula or equation, just give me the count, density and the most concentrated region. I will not give the bounding box width and height, whatever you want calculate by yourself. Give me the content in a para of 7 lines without points and include some randomness. If there is no content, then add some more information about the image. your max tokens are 150 dont exceed that."

    if message:
        messages.append(
            {"role": "user", "content": message},
        )
        chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages,max_tokens= 110)
    reply = chat.choices[0].message.content
    print(f"ChatGPT: {reply}")
    messages.append({"role": "assistant", "content": reply})

    return reply 



def city_district_state(Latitude,Longitude):
    geolocator = Nominatim(user_agent="aqua-app")
    location = geolocator.reverse(Latitude+","+Longitude)
    address = location.raw['address']
    # traverse the data
    city = address.get('city', '')
    state_district = address.get('state_district', '')
    country = address.get('country', '')
    suburb = address.get('suburb','')
    city_district = address.get('city_district','')
    state = address.get('state','')
    country = address.get('country','')
    if city == '':
        city = suburb
    if state_district == '':
        state_district = city_district
    if state == '':
        state = state_district.split(" ")[0]
    return city, state_district, state,country

def plastic_area(bounding_boxes):
    total_area = 0

    for box in bounding_boxes:
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        area = width * height
        total_area += area
    return total_area



def get_relative_position(latitude, longitude, country_name):
    geolocator = Nominatim(user_agent="geo_locator")

    country_location = geolocator.geocode(country_name)

    if country_location is None:
        return "Country location not found"

    country_lat, country_lon = country_location.latitude, country_location.longitude

    # Convert latitude to float
    latitude = float(latitude)
    longitude = float(longitude)
    distance = geodesic((latitude, longitude), (country_lat, country_lon)).kilometers

    if latitude > country_lat:
        latitude_position = "north"
    else:
        latitude_position = "south"

    if longitude > country_lon:
        longitude_position = "east"
    else:
        longitude_position = "west"

    return f"{latitude_position}-{longitude_position}"


def get_weather_and_region(country, latitude, longitude):
    api_key = os.getenv('weather_api')

    # Extract state name from address
  
    # Step 2: Form API Request to OpenWeather

    open_weather_url = f'https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid=cffecceafab4795d98bde4086c0332bb'

    # Step 3: Get Weather Data

    response = requests.get(open_weather_url)
    weather_data = response.json()

    # Extract specific weather and temperature values
    extracted_weather = {
        "weather": weather_data['weather'][0]['main'].lower(),
        "temp": round(weather_data['main']['temp'] - 273.15, 1),  # Convert Kelvin to Celsius and round to 1 decimal
        "region": get_relative_position(latitude,longitude,country).lower()
    }

    return extracted_weather

# Check path is a video

video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"]

def is_video_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    
    return file_extension in video_extensions #True or False


# Replace with your actual API key and file path
def cloud_image(image):
    # Define the API endpoint URL
    api_key = os.getenv('FILE_STACK_API')

    url = "https://www.filestackapi.com/api/store/S3?key=" + api_key

    # Set the headers for the request
    headers = {
        "Content-Type": "image/png"
    }

    # Read the file as binary data and prepare it for upload
   

    # Make the POST request to upload the file
    response = requests.post(url, data=image, headers=headers)

    # Check the response status and content
    if response.status_code == 200:
        response_json = response.json()  # Parse the JSON response
        print("File uploaded successfully.")
        return response_json['url']
    else:
        print("File upload failed. Status code:", response.status_code)
        print("Response:", response.text)

def cloud_video(video):
    # Define the API endpoint URL
    api_key = os.getenv('FILE_STACK_API')

    url = "https://www.filestackapi.com/api/store/S3?key=" + api_key

    # Set the headers for the request
    headers = {
        "Content-Type": "video/mp4"
    }

    # Read the file as binary data and prepare it for upload
    with open(video, "rb") as file:
        file_data = file.read()

    # Make the POST request to upload the file
    response = requests.post(url, data=file_data, headers=headers)

    # Check the response status and content
    if response.status_code == 200:
        response_json = response.json()  # Parse the JSON response
        print("File uploaded successfully.")
        return response_json['url']
    else:
        print("File upload failed. Status code:", response.status_code)
        print("Response:", response.text)

def draw_boxes_on_image(image_path, boxes, scores, labels):
    img = cv2.imread(image_path)
    
    for box, score, label in zip(boxes, scores, labels):
        box = box.astype(int)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (128, 0, 128), 4)  # Purple color and adjusted thickness
        text = f"{label}: {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1  # Increase this value to increase font size
        font_thickness = 2
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = box[0]
        text_y = box[1] - 10  # Adjust the vertical position of the text

        # Draw a filled purple rectangle as background for the text
        cv2.rectangle(img, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), (128, 0, 128), cv2.FILLED)
        
        # Draw the text in white color
        cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)
        _, image_data = cv2.imencode('.jpg', img)
        cloud_url = cloud_image(image_data.tobytes())
        return cloud_url


#----------------------------------------------------------------------------------------------------------------------------

def login_required(f):
  @wraps(f)
  def wrap(*args, **kwargs):
    if 'logged_in' in session:
      return f(*args, **kwargs)
    else:
      return redirect('/')
  
  return wrap

# Routes
class User:

  def start_session(self, user):
    del user['password']
    session['logged_in'] = True
    session['user'] = user
    print("Session data:", session)  # Add this line
    return jsonify(user), 200

  def signup(self):
    print(request.form)

    # Create the user object
    user = {
      "_id": uuid.uuid4().hex,
      "name": request.form.get('name'),
      "email": request.form.get('email'),
      "password": request.form.get('password'),
      "role": "user"
    }

    # Encrypt the password
    user['password'] = pbkdf2_sha256.encrypt(user['password'])

    # Check for existing email address
    if db.users.find_one({ "email": user['email'] }):
      return jsonify({ "error": "Email address already in use" }), 400

    if db.users.insert_one(user):
      return self.start_session(user)

    return jsonify({ "error": "Signup failed" }), 400
  
  def signout(self):
    session.clear()
    return redirect('/')
  
  def login(self):

    user = db.users.find_one({
      "email": request.form.get('email')
    })
    print("User data:", user)  # Add this line

    if user and pbkdf2_sha256.verify(request.form.get('password'), user['password']):
      return self.start_session(user)
    
    return jsonify({ "error": "Invalid login credentials" }), 401

# ----------------------------------------------------------------------------------SELECTED FILES-------------------------------->

@app.route('/upload/selected_files/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER_1'], filename)

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    user_id = session['user']['_id']
    role = session['user']['role']
    print(role)
    current_time = datetime.now().strftime('%Y-%m-%d')
    print(request)
    if 'media' in request.files:
        media = request.files['media']
        address = request.form['address']
        latitude = request.form['latitude']
        longitude = request.form['longitude']
        mapLink = request.form['mapLink']
        print(latitude,longitude)
        city, state_district, state ,country= city_district_state(latitude,longitude)
        result = get_weather_and_region(country,latitude,longitude)
        weather = result['weather']
        temperature = result['temp']
        region = result['region']
    
        if media.filename != '':

            media_path = os.path.join(app.config['UPLOAD_FOLDER_1'], media.filename)
            media.save(media_path)
            print("Saved path",media_path)
            # Perform object detection
            if is_video_file(media_path):

                video_file_name =  inference_video(media.filename,media_path)
                video_inference_path = cloud_video(video_file_name)
                print(video_file_name)
                explanation = "Explainable AI is not available for video"
                response_data = {
                    'image_url': video_inference_path,
                    'explanation': explanation,
                    'isVideo': True
                }
                
                return jsonify(response_data)
            else:
                print(f"{media_path} is not a video file.")
            
          

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Submit each function for parallel execution
                    multi_class_future = executor.submit(predict_rcnn_multi, media_path)
                    ensemble_future = executor.submit(ensemble_singleClass, media_path)
                    moving_stat_future = executor.submit(moving_stat, media_path)

                    # Wait for all parallel tasks to complete
                    concurrent.futures.wait([multi_class_future, ensemble_future, moving_stat_future])

                    # Get the results from the futures
                    filtered_boxes, filtered_scores, mapped_names = multi_class_future.result()
                    ensembled_detections, conf, lab = ensemble_future.result()
                    boxes, probs, mapped_labels = moving_stat_future.result()
            #  

                mutli_class_link = draw_boxes_on_image(media_path,filtered_boxes,filtered_scores,mapped_names)
                mov_stat_link = draw_boxes_on_image(media_path,boxes,probs,mapped_labels)

                calculated_plastic_area = plastic_area(ensembled_detections)
                count_of_plastics = len(ensembled_detections)

                #explainable AI
                # send_alert(float(latitude),float(longitude),city,count_of_plastics,mapLink)
                # explanation = generate_explanations(ensembled_detections)
                with concurrent.futures.ThreadPoolExecutor() as executor:
                        executor.submit(send_alert, float(latitude), float(longitude), city, count_of_plastics, mapLink)
                        explanation_future = executor.submit(generate_explanations, ensembled_detections)

                explanation = explanation_future.result()


                # Convert PIL image to OpenCV format
                image = Image.open(media_path)
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                # Loop through the coordinates and draw bounding boxes
                for coord in ensembled_detections:
                    x_min, y_min, x_max, y_max = map(int, coord)
                    cv2.rectangle(image_cv, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                    label = "plastic"
                    cv2.putText(image_cv, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

                # Save the labeled image
                _, image_data = cv2.imencode('.jpg', image_cv)
                cloud_url = cloud_image(image_data.tobytes())
                predicted_image_url = cloud_url
                

                response_data = {
                    'image_url': predicted_image_url,
                    'multi_class':mutli_class_link,
                    'mov_stat':mov_stat_link,
                    'explanation': explanation
                }

                collection = db['metrics']

                new_prediction = {
                'user_id': user_id,
                'coordinates': ensembled_detections,
                'time': current_time,
                'latitude' : latitude,
                'longitude' : longitude,
                'count':count_of_plastics,
                'plastic_area': calculated_plastic_area,
                'city':city,
                'district':state_district,
                'state':state,
                'weather':weather,
                'temperature':temperature,
                'region':region
                }
            
                authority_data ={
                "city" : city,
                "state" : state,
                "geotag" : mapLink,
                "time" : current_time,
                "plastic_count" : count_of_plastics,
                "predicted_image" : predicted_image_url,
                "explain_ai_content" : explanation,
                "status" : "pending"
            }
            authority_collection.insert_one(authority_data)
            collection.insert_one(new_prediction)


            os.remove(media_path)
            return jsonify(response_data)

    return "No media selected for upload."

# ----------------------------------------------------------------------------------TAKEN FILES-------------------------------->

@app.route('/upload/taken_files/<filename>')
def uploaded_file_2(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER_2'], filename)

@app.route('/uploadPhoto', methods=['POST'])
@login_required
def uploadPhoto():
    user_id = session['user']['_id']
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if 'mapLink' in request.form:
        map_link = request.form['mapLink']
    else:
        map_link = None

    if 'address' in request.form:
        address = request.form['address']
    else:
        address = None
    if 'latitude' in request.form:
        latitude = request.form['latitude']
    else:
        latitude = None
    if 'longitude' in request.form:
        longitude = request.form['longitude']
    else:
        longitude = None
    city, state_district, state,country = city_district_state(latitude,longitude)
    result = get_weather_and_region(country, latitude, longitude)
    weather = result['weather']
    temperature = result['temp']
    region = result['region']
    global captured_image
    if 'image' in request.files:
        image_file = request.files['image']
        if image_file.filename != '':
            image_path = os.path.join(app.config['UPLOAD_FOLDER_2'], secure_filename(image_file.filename))
            image_file.save(image_path)


            filtered_boxes, filtered_scores ,mapped_names=predict_rcnn_multi(image_path)
            mutli_class_link = draw_boxes_on_image(image_path,filtered_boxes,filtered_scores,mapped_names)


            
            boxes,probs,mapped_labels =  moving_stat(image_path)

            mov_stat_link = draw_boxes_on_image(image_path,boxes,probs,mapped_labels)
            
            ensembled_detections, conf, lab = ensemble_singleClass(image_path)

           
            calculated_plastic_area = plastic_area(ensembled_detections)
            count_of_plastics = len(ensembled_detections)
            send_alert(float(latitude),float(longitude),city,count_of_plastics,map_link)

            #explainable AI
            # plasticlocation = geotag(image_path)
            explanation = generate_explanations(ensembled_detections)


            # Convert PIL image to OpenCV format
            image = Image.open(image_path)
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Loop through the coordinates and draw bounding boxes
            for coord in ensembled_detections:
                x_min, y_min, x_max, y_max = map(int, coord)
                cv2.rectangle(image_cv, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                label = "plastic"
                cv2.putText(image_cv, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            # Save the labeled image
            _, image_data = cv2.imencode('.jpg', image_cv)
            cloud_url = cloud_image(image_data.tobytes())
            predicted_image_url = cloud_url
            
            response_data = {
                'image_url': predicted_image_url,
                'multi_class':mutli_class_link,
                'mov_stat':mov_stat_link,
                  'explanation': explanation,
                  'location':map_link
            }
            collection = db['metrics']

            new_prediction = {
            'user_id': user_id,
            'coordinates': ensembled_detections,
            'time': current_time,
            'latitude' : latitude,
            'longitude' : longitude,
            'count':count_of_plastics,
            'plastic_area': calculated_plastic_area,
            'city':city,
            'district':state_district,
            'state':state,
            'weather':weather,
            'temperature':temperature,
            'region':region
            }
            authority_data={
                "city" : city,
                "state" : state,
                "geotag" : map_link,
                "time" : current_time,
                "plastic_count" : count_of_plastics,
                "predicted_image" : predicted_image_url,
                "explain_ai_content" : explanation,
                "status" : "pending"
            }
            authority_collection.insert_one(authority_data)
            collection.insert_one(new_prediction)
            os.remove(image_path)

            return jsonify(response_data)

    return "No media selected for upload."

# -------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------
# Signup mand login 


  

@app.route('/user/signup', methods=['POST'])
def signup():
  return User().signup()

@app.route('/user/signout')
def signout():
  return User().signout()

@app.route('/user/login', methods=['POST'])
def login():
  return User().login()


@app.route('/')
def home():
  return render_template('home.html')

@app.route('/dashboard/')
@login_required
def dashboard():
  print("Session data:", session)  # Add this line

  return render_template('index.html')


@app.route('/post_new', methods=['POST'])
@login_required
def post_new():
    print("inside post new",session)
#    return "success"
    if 'user' in session:
        print("i am in")
        user_id = session['user']['_id']
        print("User ID:", user_id)  # Add this line

        content = request.json.get('content')
        post_id = uuid.uuid4().hex  # Generate a random post ID
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        new_post = {
            'post': post_id,
            'user_id': user_id,
            'content': content,
            'time': current_time,
            'like_count': 0,  # Initialize like count to 0
            'comment_count': 0  ,# Initialize comment count to 0
            'liked_by': []    # Array to store user IDs who liked the post        
            }
        
        # Insert the new post into the database
        db.post.insert_one(new_post)
        
        return jsonify({'message': 'Post created successfully'}), 201
    else:
        return jsonify({'error': 'User not logged in'}), 401
    
# ... Your other imports and app setup ...

def get_time_ago(post_time_str):
    post_time = datetime.strptime(post_time_str, '%Y-%m-%d %H:%M:%S')
    current_time = datetime.now()
    time_difference = current_time - post_time

    if time_difference < timedelta(minutes=1):
        return "Just now"
    elif time_difference < timedelta(hours=1):
        return f"{int(time_difference.seconds // 60)} minutes ago"
    elif time_difference < timedelta(days=1):
        return f"{int(time_difference.seconds // 3600)} hours ago"
    elif time_difference < timedelta(days=30):
        return f"{int(time_difference.days)} days ago"
    else:
        return post_time.strftime('%b %d, %Y')  # Return the full date if more than 30 days ago
    
@app.route('/get_posts_and_comments', methods=['GET'])
@login_required
def get_posts_and_comments():
    # Fetch posts from the MongoDB collection
    posts_collection = db['post']  # Replace 'posts' with your collection name
    posts = list(posts_collection.find())
    user_id = session['user']['_id']
    
    # Fetch usernames from the 'users' collection and create a dictionary
    users_collection = db['users']  # Replace 'users' with your collection name
    users_dict = {}
    for user in users_collection.find():
        users_dict[user['_id']] = user['name']

    # Convert posts to a list of dictionaries with usernames
    posts_data = [
        {
            "postID": post['post'],  # Assuming the field name is 'postID' for post IDs
            "username": users_dict.get(post['user_id'], "Unknown User"),  # Use the user ID to fetch the username
            "timeAgo": get_time_ago(post['time']),
            "content": post['content'],
            "like_count": post['like_count'],
            "comment_count": post['comment_count'],
            "liked_by": post['liked_by'],  # Include the liked_by array
            "is_liked": user_id in post['liked_by'],
            "is_owner": post['user_id'] == user_id
        }
        for post in posts
    ]
    posts_data = sorted(posts_data, key=lambda x: x['timeAgo'], reverse=True)

    return jsonify({"posts": posts_data})

@app.route('/like_post', methods=['POST'])
@login_required
def like_post():
    data = request.get_json()
    print(data)
    post_id = data.get('postID')
    # Replace this with the actual current user's ID
    user_id = session['user']['_id']

    posts_collection = db['post']
    post = posts_collection.find_one({'post': post_id})
    print(post)
    if user_id in post['liked_by']:
        # User already liked the post, remove like
        posts_collection.update_one(
            {'post': post_id},
            {'$inc': {'like_count': -1}, '$pull': {'liked_by': user_id}}
        )
        is_liked = False  # Set to False since the user is unliking the post
    else:
        # User hasn't liked the post, add like
        posts_collection.update_one(
            {'post': post_id},
            {'$inc': {'like_count': 1}, '$push': {'liked_by': user_id}}
        )
        is_liked = True  # Set to True since the user is liking the post

    like_count = post['like_count'] + 1 if is_liked else post['like_count'] - 1

    return jsonify({"like_count": like_count, "is_liked": is_liked})

@app.route('/post_comment', methods=['POST'])
@login_required
def post_comment():
    data = request.get_json()
    userID = session['user']['_id']
    postID = data.get('postID')
    comment = data.get('comment')
    commentCollection = db['comments']
    post_collection = db['post']
    post_collection.update_one(
            {'post': postID},
            {'$inc': {'comment_count': 1}}
    )
    new_comment = {
        "user_id": userID,
        "post": postID,
        "comment":comment,
        "time":  datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    commentCollection.insert_one(new_comment)
    return jsonify(success=True)


@app.route('/get_comments_for_post/<post_id>', methods=['GET'])
@login_required
def get_comments(post_id):
    comments_collection = db['comments']
    comments = list(comments_collection.find({'post': post_id}))
    comments_data = []
    users_collection = db['users']  # Replace 'users' with your collection name
    users_dict = {}
    for user in users_collection.find():
        users_dict[user['_id']] = user['name']
    comments_data = [
        {
            'user_name': users_dict.get(comment['user_id'], "Unknown User"),  # Replace with actual user data
            'content': comment['comment'],
            'timestamp':  get_time_ago(comment['time'] )
        }
        for comment in comments
    ]
    comments_data = sorted(comments_data, key=lambda x: x['timestamp'], reverse=True)
    return jsonify({"comments":comments_data})

@app.route('/delete_post', methods=['POST'])
@login_required
def delete_post():
    data = request.get_json()
    post_id = data.get('postID')
    user_id = session['user']['_id']

    posts_collection = db['post']  # Replace 'post' with your collection name
    post = posts_collection.find_one({'post': post_id})

    if post and post['user_id'] == user_id:
        # Delete the post if the current user is the owner of the post
        posts_collection.delete_one({'post': post_id})
        return jsonify(message='Post deleted successfully')
    else:
        return jsonify(error='Unauthorized or post not found'), 403
    
# Authority Dashboard------------------------------------------------

@app.route('/display_documents', methods=['GET'])
@login_required
def display_documents():
    role = session['user']['role']
    documents = list(authority_collection.find())  # Retrieve all documents from the collection
    return render_template('authority.html', documents=documents, role=role)

@app.route('/move_to_completed/<document_id>', methods=['POST'])
def move_to_completed(document_id):
    obj_id = ObjectId(document_id)
    
    # Update the status of the document to "completed" in the database
    authority_collection.update_one({"_id": obj_id}, {"$set": {"status": "completed"}})
    
    # Retrieve the updated list of pending documents from the database
    pending_documents = list(authority_collection.find({"status": "pending"}))
    
    # Convert the ObjectId to string for JSON serialization
    for doc in pending_documents:
        doc['_id'] = str(doc['_id'])
    
    return jsonify(pending_documents)

@app.route('/filter', methods=['POST'])
@login_required
def apply_filters():
    selected_min_plastic_count = request.form.get('fromInput')
    selected_max_plastic_count = request.form.get('toInput')
    selected_status = request.form.get('status')
    selected_city = request.form.get('city')
    selected_state = request.form.get('state')

    # Convert input values to integers if they are not None or empty
    if selected_min_plastic_count is not None and selected_min_plastic_count != '':
        selected_min_plastic_count = int(selected_min_plastic_count)
    else:
        selected_min_plastic_count = None

    if selected_max_plastic_count is not None and selected_max_plastic_count != '':
        selected_max_plastic_count = int(selected_max_plastic_count)
    else:
        selected_max_plastic_count = None

    # Define the filter criteria based on the selected filters
    filter_criteria = {}

    # Construct plastic count range filter
    if selected_min_plastic_count is not None and selected_max_plastic_count is not None:
        filter_criteria['plastic_count'] = {
            "$gte": selected_min_plastic_count,
            "$lte": selected_max_plastic_count
        }

    if selected_status:
        filter_criteria['status'] = selected_status
    if selected_city:
        filter_criteria['city'] = selected_city
    if selected_state:
        filter_criteria['state'] = selected_state

    # Query documents based on the filter criteria
    filtered_documents = list(authority_collection.find(filter_criteria))

    # Convert ObjectId to string for each document
    for doc in filtered_documents:
        doc['_id'] = str(doc['_id'])

    # Render the filtered documents as JSON
    return jsonify(filtered_documents)

@app.route('/get_city_options', methods=['GET'])
def get_city_options():
    city_options = list(authority_collection.distinct("city"))
    return jsonify(city_options)

@app.route('/get_state_options', methods=['GET'])
def get_state_options():
    state_options = list(authority_collection.distinct("state"))
    return jsonify(state_options)

ngo_collection = db.ngo
@app.route("/add_ngo", methods=["POST"])
def add_ngo():
    try:
        # Get data from the request
        data = request.json

        # Insert the NGO document into the collection
        result = ngo_collection.insert_one(data)

        # Respond with a success message and the inserted document's ID
        response = {"message": "NGO added successfully", "id": str(result.inserted_id)}
        return jsonify(response), 201

    except Exception as e:
        # Handle errors and respond with an error message
        response = {"error": str(e)}
        return jsonify(response), 500
    
mode = "dev"
if __name__ == '__main__':
    if mode == "dev":
        app.run( host='0.0.0.0',port=5000, debug= True)
    else:
        print("http://localhost:50100")
        serve(app, host='0.0.0.0',port=50100, threads=8)