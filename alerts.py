import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from twilio.rest import Client
from math import radians, sin, cos, sqrt, atan2
import pymongo
from dotenv import load_dotenv
import os

load_dotenv()



MAX_DISTANCE_THRESHOLD = 25 

def fetch_ngos_from_database():
    try:
        # Replace with your MongoDB connection details
        client = pymongo.MongoClient("mongodb+srv://aqua:12345@cluster0.n99hlyd.mongodb.net/?retryWrites=true&w=majority")
        db = client["connect"]
        collection = db["ngo"]

        # Fetch NGO data
        ngos = list(collection.find({}, {"name": 1, "email": 1, "phone_number": 1, "latitude": 1, "longitude": 1}))
        
        return ngos
    
    except Exception as error:
        print("Error fetching data from MongoDB:", error)
        return []

def calculate_distance(coord1, coord2):
    # Approximate radius of the Earth in km
    R = 6371.0

    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

def find_nearest_ngo(user_location, ngos):
    nearest_ngo = None
    min_distance = MAX_DISTANCE_THRESHOLD  # Use the predefined threshold value

    for ngo in ngos:
        ngo_location = (ngo["latitude"], ngo["longitude"])
        distance = calculate_distance(user_location, ngo_location)
        if distance < min_distance:
            min_distance = distance
            nearest_ngo = ngo

    return nearest_ngo

def send_email_to_ngo(user_location, ngo_location, ngo_name, ngo_email, message_content):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    smtp_username = "aquavision.connect@gmail.com"
    smtp_password = "txecmalgedgrwvjl"

    distance = calculate_distance(user_location, ngo_location)
    
    if distance < MAX_DISTANCE_THRESHOLD:
        subject = "🌍 Environmental Alert: Plastic Waste Detected"
        body = f"""
            <html>
            <head>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        background-color: #f6f6f6;
                    }}
                    .container {{
                        max-width: 600px;
                        margin: 0 auto;
                        padding: 20px;
                        background-color: white;
                        border: 1px solid black;
                        border-radius: 10px;
                    }}
                    .logo {{
                        margin: 10px 0px;
                        text-align: center;
                    }}
                    .logo img {{
                        max-width: 80%; /* Adjust the image width as needed */
                        height: auto;
                    }}
                    .content {{
                        padding: 20px;
                    }}
                    .details {{
                        margin-top: 20px;
                        font-size: 14px;
                    }}
                    .details-table {{
                        width: 100%;
                        margin-top: 10px;
                    }}
                    .details-table td {{
                        padding: 5px 0;
                    }}
                    .footer {{
                        margin-top: 20px;
                        text-align: center;
                        color: #888;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="logo">
                        <img src="https://res.cloudinary.com/dngzcanli/image/upload/v1692784769/avlogo-new_qq9d1t.jpg" alt="Your Logo">
                    </div>
                    <div class="content">
                        <h1 style="color: #007BFF;">🌱 Plastic Waste Detected</h1>
                        <p>Hello {ngo_name} 👋,</p>
                        <p>We have detected plastic waste in your vicinity 🌍. Here are the details:</p>
                        <table class="details-table">
                            <tr>
                                <td><strong>Area</strong></td>
                                <td>{message_content["Area"]}</td>
                            </tr>
                            <tr>
                                <td><strong>Count</strong></td>
                                <td>{message_content["Count"]}</td>
                            </tr>
                            <tr>
                                <td><strong>Location</strong></td>
                                <td>{message_content["Location"]}</td>
                            </tr>
                        </table>
                        <p>Thank you for your commitment to plastic waste management and environmental conservation. Together, we can make a difference!</p>
                    </div>
                    <div class="footer">
                        <p>Best regards,<br>Team AquaVision</p>
                    </div>
                </div>
            </body>
            </html>
        """
        
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = "aquavision.connect@example.com"
        msg["To"] = ngo_email        
        msg.attach(MIMEText(body, "html"))
        
        # Sending the email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.sendmail(msg["From"], msg["To"], msg.as_string())


def send_twilio_message(to_phone_number, message_content):
    # Your Twilio Account SID and Auth Token
    account_sid = os.getenv('twilio_account_sid')
    auth_token = os.getenv('twilio_auth_token')

    # Initialize the Twilio client
    client = Client(account_sid, auth_token)

    try:
        # Send an SMS message
        message_body = f"Area: {message_content['Area']}\nCount: {message_content['Count']}\nLocation: {message_content['Location']}"

        # Send the message with the formatted message body
        message = client.messages.create(
            body=message_body,
            from_='+12565026139',  # Your Twilio phone number
            to=to_phone_number  # The recipient's phone number
        )

        # Print the message SID (optional)
        print(f"Message SID: {message.sid}")

        # Return True to indicate successful message sending
        return True

    except Exception as e:
        print(f"Error sending Twilio message: {str(e)}")
        return False





def send_notification(user_location, area, count, geotag):
    # Fetch NGO data from the database
    ngos = fetch_ngos_from_database()

    # Find the nearest NGO based on user's location
    nearest_ngo = find_nearest_ngo(user_location, ngos)

    if nearest_ngo:
        message_content = {
            "Area": area,
            "Count": count,
            "Location": geotag  # Replace with image details
        }
        send_email_to_ngo(user_location, (nearest_ngo["latitude"], nearest_ngo["longitude"]), nearest_ngo["name"], nearest_ngo["email"], message_content)
        send_twilio_message('+918525091777', message_content)
    else:
        print("No nearby NGOs found")

def send_alert(latitude,longitude,area,count,geotag):
    # Example data for the user
    user_location = (latitude, longitude)  # Chennai (Central)
   
    # Send notification
    send_notification(user_location, area, count, geotag)

# send_alert(13.0225,80.1728,"chennai","23","http://summa.con")