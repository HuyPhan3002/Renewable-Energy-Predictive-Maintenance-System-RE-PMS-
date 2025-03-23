import os
import numpy as np
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# Define dataset path
data_path = r"Traning_data_set" # Modify your training data set

# Check if dataset exists
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}")

# Load dataset
df = pd.read_csv(data_path)

# Handle missing values by imputing with the median
df.fillna(df.median(), inplace=True)

# Split into features (X) and target (y)
X = df.drop(columns=["Target","ID"])
y = df["Target"]

# Split into training (80%) and validation (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on validation set
y_pred = rf_model.predict(X_test)

# Identify failure cases (where prediction = 1)
failure_cases = X_test[y_pred == 1]
failure_indice = X_test.index[y_pred == 1]
failure_ID = df.loc[failure_indice, "ID"].tolist()

# Email credentials
# SMTP domain name: smtp.gmail.com, Yahoo Mail: smtp.mail.yahoo.com, Outlook.com/Hotmail: smtp-mail.outlook.com
SMTP_SERVER = "smtp.gmail.com" 
SMTP_PORT = 587
EMAIL_SENDER = "sender_email@gmail.com" # Modify sender email
EMAIL_PASSWORD = "password" # Modify password
EMAIL_RECEIVER = "receiver@gmail.com" # Modify receiver email

def send_email_alert(failure_count):
    """Send an email if failure is predicted."""
    subject = "⚠️ Predictive Maintenance Alert: Potential Failures Detected"
    sorted_failure_id = sorted(failure_ID)
    
    # Format failure IDs as a numbered list
    failure_list = "\n".join([f"{i+1}. Machine ID: {failure}" for i, failure in enumerate(sorted_failure_id)])

    body = f"""
    ALERT: Predictive Maintenance System

    Our AI model has detected {failure_count} machines at risk of failure.
    Immediate attention is required to prevent downtime.

    Affected Machines:
    ------------------
{failure_list}

    Please review the data and take necessary maintenance actions.

    Regards,  
    Predictive Maintenance System
    """

    msg = MIMEMultipart()
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))  # Plain text format

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()
        print("Email alert sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")

# If failures are detected, send an email
failure_count = len(failure_cases)
if failure_count > 0:
    send_email_alert(failure_count)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print results
print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)