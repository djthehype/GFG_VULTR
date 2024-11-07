import streamlit as st
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import hashlib
import json
from time import time

import streamlit as st

# Load CSS styles
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load pr_3.css
load_css("pr_3.css")


#st.markdown('<div class="doctor-info">Doctor Information:</div>', unsafe_allow_html=True)
#st.markdown('<div class="success">Account created successfully! Please log in.</div>', unsafe_allow_html=True)



# Connect to SQLite database
conn = sqlite3.connect('form_2.db', check_same_thread=False, timeout=10)
cursor = conn.cursor()

# Create tables for user registration and blockchain data if they don't exist
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS registrations (
        username TEXT(50),
        password TEXT(50),
        surname TEXT(50),
        name TEXT(50),
        age TEXT(50),
        country TEXT(50),
        birthcountry TEXT(50)
    )
    """
)

cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS blockchain_data (
        block_index INTEGER,
        block_hash TEXT,
        block_data TEXT
    )
    """
)

conn.commit()

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv(r"C:\Users\Faizan\Desktop\Seminar\Disease_symptom_and_patient_profile_dataset.csv")
    return data

data = load_data()

# Data Preprocessing
def preprocess_data(data):
    label_encoder = LabelEncoder()

    binary_columns = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Gender', 'Outcome Variable']

    for column in binary_columns:
        data[column] = label_encoder.fit_transform(data[column])

    data = pd.get_dummies(data, columns=['Blood Pressure', 'Cholesterol Level'], drop_first=True)
    return data

processed_data = preprocess_data(data)

X = processed_data.drop(columns=['Disease'])
y = processed_data['Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@st.cache_data
def train_model(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

model = train_model(X_train, y_train)

# Patient Details Form
def patient_details_form():
    st.subheader("Patient Details Form")

    name = st.text_input("Enter your Name")
    age = st.slider("Enter your Age", 1, 100, 30)
    gender = st.selectbox("Gender", ['Male', 'Female'])
    address = st.text_area("Enter your Address")

    return {"Name": name, "Age": age, "Gender": gender, "Address": address}

# Function for user input features (symptoms)
def user_input_features():
    st.subheader("Symptom Checker")

    col1, col2 = st.columns(2)

    with col1:
        Fever = st.selectbox('Fever', ('Yes', 'No'))
        Cough = st.selectbox('Cough', ('Yes', 'No'))
        Fatigue = st.selectbox('Fatigue', ('Yes', 'No'))
        Difficulty_Breathing = st.selectbox('Difficulty Breathing', ('Yes', 'No'))

    with col2:
        Blood_Pressure = st.selectbox('Blood Pressure', ('Low', 'Normal', 'High'))
        Cholesterol_Level = st.selectbox('Cholesterol Level', ('Low', 'Normal', 'High'))

    data = {
        'Fever': 1 if Fever == 'Yes' else 0,
        'Cough': 1 if Cough == 'Yes' else 0,
        'Fatigue': 1 if Fatigue == 'Yes' else 0,
        'Difficulty Breathing': 1 if Difficulty_Breathing == 'Yes' else 0,
        'Blood Pressure_Low': 1 if Blood_Pressure == 'Low' else 0,
        'Blood Pressure_Normal': 1 if Blood_Pressure == 'Normal' else 0,
        'Cholesterol Level_Low': 1 if Cholesterol_Level == 'Low' else 0,
        'Cholesterol Level_Normal': 1 if Cholesterol_Level == 'Normal' else 0
    }

    features = pd.DataFrame(data, index=[0])
    missing_cols = set(X_train.columns) - set(features.columns)
    for col in missing_cols:
        features[col] = 0
    features = features[X_train.columns]

    return features

# Blockchain Implementation
class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = json.dumps(self.__dict__, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = []
        self.create_block(0, "0")  # Create the genesis block

    def create_block(self, index, previous_hash):
        block = Block(index, time(), {}, previous_hash)
        self.chain.append(block)
        return block

    def add_patient_data(self, patient_data):
        last_block = self.chain[-1]
        index = last_block.index + 1
        previous_hash = last_block.hash
        block = self.create_block(index, previous_hash)
        block.data = patient_data  # Add patient data to the new block
        
        # Save block data to the database
        cursor.execute("INSERT INTO blockchain_data (block_index, block_hash, block_data) VALUES (?, ?, ?)",
                       (block.index, block.hash, json.dumps(block.data)))
        conn.commit()

        return block

# Initialize blockchain
blockchain = Blockchain()

def get_doctor_for_disease(disease):
    disease_to_doctor = {
        'Influenza': {'Name': 'Dr. Smith', 'Specialty': 'General Practitioner', 'Clinic': 'City Health Clinic', 'Address': '123 Main St'},
        'Common Cold': {'Name': 'Dr. Smith', 'Specialty': 'General Practitioner', 'Clinic': 'City Health Clinic', 'Address': '123 Main St'},
        'Pneumonia': {'Name': 'Dr. Johnson', 'Specialty': 'Pulmonologist', 'Clinic': 'Breath Easy Clinic', 'Address': '456 Elm St'},
        'Bronchitis': {'Name': 'Dr. Johnson', 'Specialty': 'Pulmonologist', 'Clinic': 'Breath Easy Clinic', 'Address': '456 Elm St'},
        'Eczema': {'Name': 'Dr. Lee', 'Specialty': 'Dermatologist', 'Clinic': 'Skin Care Center', 'Address': '789 Oak St'},
        'Psoriasis': {'Name': 'Dr. Lee', 'Specialty': 'Dermatologist', 'Clinic': 'Skin Care Center', 'Address': '789 Oak St'},
    }

    return disease_to_doctor.get(disease, None)

def display_doctor_info(doctor):
    if doctor:
        st.write(f"**Doctor Name:** {doctor['Name']}")
        st.write(f"**Specialty:** {doctor['Specialty']}")
        st.write(f"**Clinic:** {doctor['Clinic']}")
        st.write(f"**Clinic Address:** {doctor['Address']}")
    else:
        st.write("No doctor recommendation available for the predicted disease.")

# Authentication
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def signup():
    st.subheader("Create a New Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Signup"):
        cursor.execute("INSERT INTO registrations (username, password) VALUES (?, ?)", 
                       (username, hash_password(password)))
        conn.commit()
        st.success("Account created successfully! Please log in.")


def login():
    st.subheader("Login to Your Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        cursor.execute("SELECT * FROM registrations WHERE username = ? AND password = ?", 
                       (username, hash_password(password)))
        user = cursor.fetchone()
        if user:
            st.session_state.authenticated = True
            st.success(f"Welcome, {username}!")
        else:
            st.error("Invalid credentials")
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        st.write("You are logged in!")
    # Main app functionalities
    else:
        st.write("Please log in.")


def logout():
    st.session_state.authenticated = False
    st.success("You have been logged out.")

# Display blockchain data
def display_blockchain_data():
    st.subheader("Blockchain Data:")
    cursor.execute("SELECT * FROM blockchain_data")
    records = cursor.fetchall()

    if records:
        for record in records:
            st.write(f"**Block Index:** {record[0]}")
            st.write(f"**Block Hash:** {record[1]}")
            st.write(f"**Block Data:** {json.loads(record[2])}")  # Convert JSON string back to dictionary
            st.write("----------")
    else:
        st.write("No blockchain data available.")

# Main Application
import streamlit as st

def app():
    # Title and Welcome Image
    st.title('Welcome to :violet[Data Dudes]')
    st.image(r"C:\Users\Faizan\Desktop\Seminar\a doctor.png", width=500)

    # User Authentication Status
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    # If user is authenticated, display prediction form and blockchain-related sections
    if st.session_state.authenticated:
        st.write("Please enter your details and symptoms to get a prediction:")

        # Collect patient details and input features for disease prediction
        patient_info = patient_details_form()  # Function to get patient details
        features = user_input_features()  # Function to get features for prediction

        if st.button("Predict Disease"):
            # Simulate prediction from a model
            prediction = model.predict(features)[0]
            st.success(f"Predicted Disease: {prediction}")

            # Add patient data to blockchain
            blockchain_data = {
                "Name": patient_info['Name'],
                "Age": patient_info['Age'],
                "Gender": patient_info['Gender'],
                "Address": patient_info['Address'],
                "Predicted Disease": prediction
            }
            blockchain.add_patient_data(blockchain_data)

            # Retrieve doctor information based on prediction
            doctor = get_doctor_for_disease(prediction)
            display_doctor_info(doctor)

        # Logout Button
        if st.button("Logout"):
            logout()  # Function to log out the user

        # Display Blockchain Data
        ##display_blockchain_data()  # Function to display stored blockchain records

    # If user is not authenticated, show login/signup options
    else:
        st.write("Please log in or sign up to continue.")
        choice = st.selectbox('Login/Signup', ['Login', 'Sign up'])
        email = st.text_input('Email Address')
        password = st.text_input('Password', type='password')

        if choice == 'Sign up':
            username = st.text_input("Enter your unique username")
            if st.button('Create my account'):
                # Placeholder for account creation
                st.success('Account created successfully! Please log in.')
        elif choice == 'Login':
            if st.button('Login'):
                # Placeholder for login logic
                st.session_state.authenticated = True

if __name__ == "__main__":
    app()