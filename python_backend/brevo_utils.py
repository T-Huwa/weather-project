import logging
from datetime import datetime
import calendar
import os
from dotenv import load_dotenv

# Import Brevo SDK
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException

# Configure logging
logger = logging.getLogger(__name__)
load_dotenv()

# Brevo API configuration
BREVO_API_KEY = os.getenv('BREVO_API_KEY')

# Email configuration
SENDER_EMAIL = 'emmannyoni5@gmail.com'
SENDER_NAME = 'Kasungu Weather'
BASE_URL = 'http://localhost:5000'

# Initialize Brevo SDK configuration
configuration = sib_api_v3_sdk.Configuration()
configuration.api_key['api-key'] = BREVO_API_KEY

def send_prediction_email(recipient_email, prediction):
    """
    Send weather prediction email using Brevo SDK
    
    Args:
        recipient_email (str): Recipient's email address
        prediction (dict): Weather prediction data
    """
    
    if not BREVO_API_KEY:
        logger.error("BREVO_API_KEY not set. Please set your Brevo API key as environment variable.")
        raise Exception("Brevo API key not configured")
    
    try:
        # Create API instance
        api_instance = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))
        
        # Format the month name
        month_name = calendar.month_name[prediction['month']]
        
        # Create email content
        subject = f"Weather Prediction for {month_name} {prediction['year']} - Kasungu, Malawi"
        
        # HTML email template
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Weather Prediction</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f4f4f4;
                }}
                .container {{
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 24px;
                }}
                .prediction-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                }}
                .prediction-card {{
                    background: #f8f9ff;
                    padding: 20px;
                    border-radius: 8px;
                    border-left: 4px solid #667eea;
                    text-align: center;
                }}
                .prediction-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #667eea;
                    margin: 10px 0;
                }}
                .prediction-label {{
                    color: #666;
                    font-size: 14px;
                    font-weight: 500;
                }}
                .note {{
                    background-color: #e8f2ff;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                    border-left: 4px solid #2196F3;
                }}
                .footer {{
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 1px solid #eee;
                    font-size: 12px;
                    color: #666;
                    text-align: center;
                }}
                .unsubscribe {{
                    margin-top: 20px;
                    font-size: 11px;
                    color: #999;
                }}
                .unsubscribe a {{
                    color: #667eea;
                    text-decoration: none;
                }}
                @media (max-width: 600px) {{
                    .prediction-grid {{
                        grid-template-columns: 1fr;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🌤️ Weather Prediction</h1>
                    <p>{month_name} {prediction['year']} - Kasungu, Malawi</p>
                </div>
                
                <p>Hello!</p>
                <p>Here's your monthly weather prediction for <strong>{month_name} {prediction['year']}</strong> in Kasungu, Malawi:</p>
                
                <div class="prediction-grid">
                    <div class="prediction-card">
                        <div class="prediction-value">{prediction['tmin']}°C</div>
                        <div class="prediction-label">🌡️ Minimum Temperature</div>
                    </div>
                    
                    <div class="prediction-card">
                        <div class="prediction-value">{prediction['tmax']}°C</div>
                        <div class="prediction-label">🌡️ Maximum Temperature</div>
                    </div>
                    
                    <div class="prediction-card">
                        <div class="prediction-value">{prediction['rainfall']} mm</div>
                        <div class="prediction-label">🌧️ Expected Rainfall</div>
                    </div>
                    
                    <div class="prediction-card">
                        <div class="prediction-value">{prediction['wind_speed']} m/s</div>
                        <div class="prediction-label">💨 Wind Speed</div>
                    </div>
                    
                    <div class="prediction-card">
                        <div class="prediction-value">{prediction['humidity']}%</div>
                        <div class="prediction-label">💧 Humidity</div>
                    </div>
                </div>
                
                <div class="note">
                    <strong>📝 Note:</strong> These predictions are based on advanced machine learning models and historical weather data. 
                    While we strive for accuracy, weather patterns can be unpredictable. Please use this information as a general guide 
                    for planning purposes.
                </div>
                
                <div class="footer">
                    <p>Thank you for subscribing to our weather prediction service!</p>
                    <p>Generated on {datetime.now().strftime("%B %d, %Y at %I:%M %p")}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Plain text version for email clients that don't support HTML
        text_content = f"""
Weather Prediction for {month_name} {prediction['year']} - Kasungu, Malawi

Hello!

Here's your monthly weather prediction for {month_name} {prediction['year']} in Kasungu, Malawi:

🌡️ Temperature Range: {prediction['tmin']}°C - {prediction['tmax']}°C
🌧️ Expected Rainfall: {prediction['rainfall']} mm
💨 Wind Speed: {prediction['wind_speed']} m/s
💧 Humidity: {prediction['humidity']}%

Note: These predictions are based on advanced machine learning models and historical weather data. 
While we strive for accuracy, weather patterns can be unpredictable. Please use this information 
as a general guide for planning purposes.

Thank you for subscribing to our weather prediction service!
Generated on {datetime.now().strftime("%B %d, %Y at %I:%M %p")}
        """
        
        # Create sender object
        sender = sib_api_v3_sdk.SendSmtpEmailSender(name=SENDER_NAME, email=SENDER_EMAIL)
        
        # Create recipient object
        to = [sib_api_v3_sdk.SendSmtpEmailTo(email=recipient_email)]
        
        # Create email object
        send_smtp_email = sib_api_v3_sdk.SendSmtpEmail(
            to=to,
            sender=sender,
            subject=subject,
            html_content=html_content,
            text_content=text_content,
            tags=["weather-prediction", f"month-{prediction['month']}", f"year-{prediction['year']}"]
        )

        print("Sending Email via Brevo SDK...")
        print(f"To: {recipient_email}")
        print(f"Subject: {subject}")
        
        # Send the email
        api_response = api_instance.send_transac_email(send_smtp_email)
        
        print("Response from Brevo SDK:")
        print(api_response)
        
        if api_response and hasattr(api_response, 'message_id'):
            logger.info(f"Successfully sent prediction email to {recipient_email}. Message ID: {api_response.message_id}")
            return True
        else:
            logger.info(f"Email sent to {recipient_email} but no message ID returned")
            return True
            
    except ApiException as e:
        logger.error(f"Brevo API Exception when sending email to {recipient_email}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error sending email to {recipient_email}: {str(e)}")
        raise e

def send_welcome_email(recipient_email):
    """
    Send welcome email to new subscribers using Brevo SDK
    
    Args:
        recipient_email (str): Recipient's email address
    """
    
    if not BREVO_API_KEY:
        logger.error("BREVO_API_KEY not set. Please set your Brevo API key as environment variable.")
        raise Exception("Brevo API key not configured")
    
    try:
        # Create API instance
        api_instance = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))
        
        subject = "Welcome to Kasungu Weather Predictions Service!"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Welcome to Kasungu Weather Predictions</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f4f4f4;
                }}
                .container {{
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                }}
                .welcome-text {{
                    font-size: 16px;
                    margin-bottom: 20px;
                }}
                .features {{
                    background: #f8f9ff;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                }}
                .features ul {{
                    margin: 0;
                    padding-left: 20px;
                }}
                .features li {{
                    margin-bottom: 8px;
                }}
                .footer {{
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 1px solid #eee;
                    font-size: 12px;
                    color: #666;
                    text-align: center;
                }}
                .unsubscribe {{
                    margin-top: 20px;
                    font-size: 11px;
                    color: #999;
                }}
                .unsubscribe a {{
                    color: #667eea;
                    text-decoration: none;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🌤️ Welcome!</h1>
                    <p>Thank you for subscribing to Kasungu Weather Predictions</p>
                </div>
                
                <div class="welcome-text">
                    <p>Hello!</p>
                    <p>Welcome to Kasungu Weather Prediction service! We're excited to have you on board.</p>
                    <p>You'll now receive monthly weather predictions for Kasungu, Malawi, delivered right to your inbox 
                    on the first day of each month.</p>
                </div>
                
                <div class="features">
                    <h3>What you'll receive:</h3>
                    <ul>
                        <li>🌡️ Monthly temperature forecasts (min/max)</li>
                        <li>🌧️ Expected rainfall predictions</li>
                        <li>💨 Wind speed forecasts</li>
                        <li>💧 Humidity predictions</li>
                        <li>📊 Data-driven insights based on machine learning models</li>
                    </ul>
                </div>
                
                <p>Our predictions are generated using advanced machine learning models trained on historical weather data 
                to provide you with accurate forecasts for planning your agricultural activities, travel, and daily life.</p>
                
                <p>Your first prediction will arrive on the 1st of next month. We hope this service helps you stay 
                prepared for the weather ahead!</p>
                
                <div class="footer">
                    <p>Thank you for trusting us with your weather information needs!</p>
                    <p>Subscribed on {datetime.now().strftime("%B %d, %Y")}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        text_content = f"""
Welcome to Weather Predictions Service!

Hello!

Welcome to our Weather Prediction service! We're excited to have you on board.

You'll now receive monthly weather predictions for Kasungu, Malawi, delivered right to your inbox 
on the first day of each month.

What you'll receive:
- Monthly temperature forecasts (min/max)
- Expected rainfall predictions  
- Wind speed forecasts
- Humidity predictions
- Data-driven insights based on machine learning models

Our predictions are generated using advanced machine learning models trained on historical weather data 
to provide you with accurate forecasts for planning your agricultural activities, travel, and daily life.

Your first prediction will arrive on the 1st of next month. We hope this service helps you stay 
prepared for the weather ahead!

Thank you for trusting us with your weather information needs!
Subscribed on {datetime.now().strftime("%B %d, %Y")}
        """
        
        # Create sender object
        sender = sib_api_v3_sdk.SendSmtpEmailSender(name=SENDER_NAME, email=SENDER_EMAIL)
        
        # Create recipient object
        to = [sib_api_v3_sdk.SendSmtpEmailTo(email=recipient_email)]
        
        # Create email object
        send_smtp_email = sib_api_v3_sdk.SendSmtpEmail(
            to=to,
            sender=sender,
            subject=subject,
            html_content=html_content,
            text_content=text_content,
            tags=["welcome-email", "subscription"]
        )
        
        # Send the email
        api_response = api_instance.send_transac_email(send_smtp_email)
        
        if api_response and hasattr(api_response, 'message_id'):
            logger.info(f"Successfully sent welcome email to {recipient_email}. Message ID: {api_response.message_id}")
            return True
        else:
            logger.info(f"Welcome email sent to {recipient_email} but no message ID returned")
            return True
            
    except ApiException as e:
        logger.error(f"Brevo API Exception when sending welcome email to {recipient_email}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error sending welcome email to {recipient_email}: {str(e)}")
        return False

def test_brevo_connection():
    """
    Test Brevo SDK connection
    """
    if not BREVO_API_KEY:
        return {"status": "error", "message": "BREVO_API_KEY not configured"}
    
    try:
        # Create API instance for account info
        api_instance = sib_api_v3_sdk.AccountApi(sib_api_v3_sdk.ApiClient(configuration))
        
        # Test connection by getting account info
        api_response = api_instance.get_account()
        
        if api_response:
            return {
                "status": "success", 
                "message": "Brevo SDK connection successful",
                "account": api_response.email if hasattr(api_response, 'email') else 'Unknown',
                "plan": api_response.plan if hasattr(api_response, 'plan') else 'Unknown'
            }
        else:
            return {"status": "error", "message": "No response from Brevo API"}
            
    except ApiException as e:
        return {"status": "error", "message": f"Brevo API Exception: {e}"}
    except Exception as e:
        return {"status": "error", "message": f"Connection failed: {str(e)}"}