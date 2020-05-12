from flask import Flask,request,jsonify
from flask_mail import Mail, Message
import sqlite3
import pandas as pd
import secrets
import string

app = Flask(__name__)

# Email configurations
app.config.update(
    DEBUG=True,
    MAIL_SERVER = 'smtp.gmail.com',
    MAIL_PORT = 465,
    MAIL_USE_SSL = True,
    MAIL_USERNAME = 'droid7developer@gmail.com',
    MAIL_PASSWORD = 'hitherebro1#')

mail = Mail(app)


@app.route('/send_email',methods=['POST'])
def send_email():
    
    # Retrieve JSON inputs
    response = request.json
    
    # Parse through JSON 
    offer_name = response['offer_name']
    
    conn = sqlite3.connect('EmployeeDatabase.db')
    
    selectQuery = 'Select Email_Id from Employee'
    cursor = conn.execute(selectQuery)
    l = []
    for row in cursor:
        l.append(row)
                
    email_df = pd.DataFrame(l, columns=['email_id'])
    
    email_recipients = list(email_df['email_id'])

    coupon_code = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for i in range(8)) 
  
    
    try:
        
        # Email Subject, Sender and Recipients
        msg = Message("Amazing Offer only for your eyes !",
                      sender = "ops@ramco.com",
                      recipients = email_recipients)
           
        # Email Body
        msg.body = """Hello Ramcoite !\nYour hard work never goes unnoticed. Here is an amazing offer for you !\nSpend your time at - """ +str(offer_name)+ """ Use code - """ +str(coupon_code)+ """\nSent with regards,\n Your Friends @ Ramco"""       
                 
        # Send Email
        mail.send(msg)
        
        return jsonify("Email Sent!")

    except Exception as e:
        print('Error Encountered:',e,flush=True)
        return jsonify('\n \n Error! Unable to send email! \n:'+str(e))
        

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port = 5023)
