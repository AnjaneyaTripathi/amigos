from flask import Flask, flash, redirect, render_template, request, session, abort, url_for
import os
from flask_mail import Mail, Message
import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import cv2
import secrets
import string
import jsonify
plt.rcParams.update({'font.size': 22})

app = Flask(__name__)

data = False


# Email configurations
app.config.update(
    DEBUG=True,
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=465,
    MAIL_USE_SSL=True,
    MAIL_USERNAME='droid7developer@gmail.com',
    MAIL_PASSWORD='hitherebro1#')

mail = Mail(app)


@app.route('/index')
def home():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        return redirect(url_for('emp'))


@app.route('/login', methods=['POST'])
def do_admin_login():
#    print('1')
    
    cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cascade_eye = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    cascade_smile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
    def detection(grayscale, img):
        
        global data
#        print('2')
    
        face = cascade_face.detectMultiScale(grayscale, 1.3, 5)
        for (x_face, y_face, w_face, h_face) in face:
            cv2.rectangle(img, (x_face, y_face), (x_face+w_face, y_face+h_face), (255, 130, 0), 2)
            ri_grayscale = grayscale[y_face:y_face+h_face, x_face:x_face+w_face]
            ri_color = img[y_face:y_face+h_face, x_face:x_face+w_face] 
            eye = cascade_eye.detectMultiScale(ri_grayscale, 1.2, 18) 
            for (x_eye, y_eye, w_eye, h_eye) in eye:
                cv2.rectangle(ri_color,(x_eye, y_eye),(x_eye+w_eye, y_eye+h_eye), (0, 180, 60), 2) 
            smile = cascade_smile.detectMultiScale(ri_grayscale, 1.7, 20)
            if smile != ():
                data = True
                print('smile')
#                print('3')
        
        return img, data
    
    vc = cv2.VideoCapture(0)
#    print('4')
    
    while True:
        _, img = vc.read() 
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        final, flag = detection(grayscale, img)
        global data
#        print('5')
    
        cv2.imshow('Video', final)
        if cv2.waitKey(1):
            if flag == True:
                print('done')
                data = False
#                print('6')
                break
    
    vc.release()
    cv2.destroyAllWindows()
    
#    if request.form['password'] == 'password' and request.form['username'] == 'admin':
    session['logged_in'] = True
    
#    else:
#        flash('wrong password!')
    
    return home()



@app.route('/emp')
def emp():
    return render_template('emp.html')


@app.route("/logout")
def logout():
    session['logged_in'] = False
    print('LOGOUT ', data)
    print('SESSION ', session['logged_in'])
    return home()


@app.route("/dashboard/<id>")
def render_dash(id):
    # function
    def happy():
        conn = sqlite3.connect('EmployeeDatabase.db')
        
        # EMPLOYEE
        
        selectQuery = 'SELECT * FROM Employee'
        cursor = conn.execute(selectQuery)
        profile = 'None'
        l = []
        for row in cursor:
            l.append(row)
        
        
        df1 = pd.DataFrame(l, columns=['emp_id', 'emp_name', 'emp_grade', 'emp_team', 'task_skills_req', 'emp_skills', 'email_id'])
        
        df1['emp_grade'] = df1['emp_grade'].apply(lambda x: x.strip())
        
        def assign_value_1(emotion):
            if emotion == 'PT':
                return 0
            elif emotion == 'G3':
                return -1
            elif emotion == 'G4':
                return -2
            elif emotion == 'G5':
                return -3
            elif emotion == 'G6':
                return -4
            elif emotion == 'G7':
                return -5
            elif emotion == 'G8':
                return -6
            elif emotion == 'G9':
                return -7
            elif emotion == 'G10':
                return -8
            elif emotion == 'G11':
                return -9
            elif emotion == 'G12':
                return -10
        
        df1['value'] = df1['emp_grade'].apply(assign_value_1)
        
        df_4 = df1[['emp_id', 'task_skills_req', 'emp_skills']]
        df_4['task_skills_req'] = df_4['task_skills_req'].apply(lambda x: x.strip())
        df_4['task_skills_req'] = df_4['task_skills_req'].apply(lambda x: x.split(', '))
        
        df_4['emp_skills'] = df_4['emp_skills'].apply(lambda x: x.strip())
        df_4['emp_skills'] = df_4['emp_skills'].apply(lambda x: x.split(', '))
        
        
        def find_common(row):
            row1 = set(row['task_skills_req'])
            row2 = set(row['emp_skills'])    
            intersec = row1.intersection(row2)        
            return len(intersec)
        
        def find_uncommon(row):
            row1 = set(row['task_skills_req'])
            row2 = set(row['emp_skills'])    
            intersec = row1.intersection(row2)        
            return len(row1) - len(intersec)
        
        
        df_4['common'] = df_4.apply(lambda row: find_common(row), axis = 1)
        
        df_4['uncommon'] = df_4.apply(lambda row: find_uncommon(row), axis = 1)
        
        
        df_4['value'] = df_4['common'] * 3 - df_4['uncommon'] * 5
        
        
        # Reaction_13745_5January
        
        selectQuery = 'SELECT * FROM Reaction_13745_5January'
        cursor = conn.execute(selectQuery)
        profile = 'None'
        l = []
        for row in cursor:
            l.append(row)
        
        
        df2 = pd.DataFrame(l, columns=['emp_id', 'emotion', 'timestamp'])
        df2['emotion'] = df2['emotion'].apply(lambda x: x.strip())
        df2['emotion'] = df2['emotion'].apply(lambda x: x.lower())
        
        df_2 = df2['emotion'].value_counts().reset_index()
        df_2.columns = ['emotion', 'count']
        
        def assign_value_2(emotion):
            if emotion == 'happy':
                return 10
            elif emotion == 'sad':
                return -8
            elif emotion == 'angry':
                return -10
            elif emotion == 'fear':
                return -5
            elif emotion == 'disgust':
                return -4
            elif emotion == 'surprise':
                return 5
            elif emotion == 'neutral':
                return 2
        
        
        df_2['value'] = df_2['emotion'].apply(assign_value_2)
        
        df_2['score'] = df_2['count'] * df_2['value'] * 0.4
        
        total_occurrences = sum(df_2['count'])
        
        df1['score'] = df1['value'] * total_occurrences * 0.05
        
        df_4['score'] = df_4['value'] * total_occurrences * 0.2
        
        
        # Reaction_13744_5January
        
        selectQuery = 'SELECT * FROM Reaction_13744_5January'
        cursor = conn.execute(selectQuery)
        profile = 'None'
        l = []
        for row in cursor:
            l.append(row)
        
        
        df3 = pd.DataFrame(l, columns=['emp_id', 'emotion', 'timestamp'])
        df3['emotion'] = df3['emotion'].apply(lambda x: x.strip())
        df3['emotion'] = df3['emotion'].apply(lambda x: x.lower())
        
        df3 = df3.drop_duplicates(subset=['timestamp'])
        
        df3['time'] = df3['timestamp'].apply(lambda x: re.sub('\d{4}-\d{2}-\d{2} ', '', x))
        
        
        # Reaction_4January
        
        selectQuery = 'SELECT * FROM Reaction_4January'
        cursor = conn.execute(selectQuery)
        profile = 'None'
        l = []
        for row in cursor:
            l.append(row)
        
        
        df4 = pd.DataFrame(l, columns=['emp_id', 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
        
        # Reaction_5January
        
        selectQuery = 'SELECT * FROM Reaction_5January'
        cursor = conn.execute(selectQuery)
        profile = 'None'
        l = []
        for row in cursor:
            l.append(row)
        
        
        df5 = pd.DataFrame(l, columns=['emp_id', 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
        
        
        # Reaction_853_5January
        
        selectQuery = 'SELECT * FROM Reaction_853_5January'
        cursor = conn.execute(selectQuery)
        profile = 'None'
        l = []
        for row in cursor:
            l.append(row)
        
        
        df6 = pd.DataFrame(l, columns=['emp_id', 'emotion', 'timestamp'])
        df6['emotion'] = df6['emotion'].apply(lambda x: x.strip())
        df6['emotion'] = df6['emotion'].apply(lambda x: x.lower())
        
        df6 = df6.drop_duplicates(subset=['timestamp'])
        
        df6['time'] = df6['timestamp'].apply(lambda x: re.sub('\d{4}-\d{2}-\d{2} ', '', x))
        
        
        
        
        # Sleep_13745_5January
        
        selectQuery = 'SELECT * FROM Sleep_13745_5January'
        cursor = conn.execute(selectQuery)
        profile = 'None'
        l = []
        for row in cursor:
            l.append(row)
        
        
        df7 = pd.DataFrame(l, columns=['emp_id', 'boredom_signs', 'timestamp'])
        
        df7 = df7.drop_duplicates()
        
        df7['time'] = df7['timestamp'].apply(lambda x: re.sub('\d{4}-\d{2}-\d{2} ', '', x))
        
        
        
        # Sleep_853_5January
        
        selectQuery = 'SELECT * FROM Sleep_853_5January'
        cursor = conn.execute(selectQuery)
        profile = 'None'
        l = []
        for row in cursor:
            l.append(row)
        
        
        df8 = pd.DataFrame(l, columns=['emp_id', 'boredom_signs', 'timestamp'])
        
        df8 = df8.drop_duplicates()
        
        df8['time'] = df8['timestamp'].apply(lambda x: re.sub('\d{4}-\d{2}-\d{2} ', '', x))
        
        
        
        # Yawn_13745_5January
        
        selectQuery = 'SELECT * FROM Yawn_13745_5January'
        cursor = conn.execute(selectQuery)
        profile = 'None'
        l = []
        for row in cursor:
            l.append(row)
        
        df9= pd.DataFrame(l, columns=['emp_id', 'boredom_signs', 'timestamp'])
        
        df9 = df9.drop_duplicates()
        
        df9['time'] = df9['timestamp'].apply(lambda x: re.sub('\d{4}-\d{2}-\d{2} ', '', x))
        
        
        df7 = df7.append(df9)
        
        df7['boredom_signs'] = df7['boredom_signs'].apply(lambda x: x.strip())
        df7['boredom_signs'] = df7['boredom_signs'].apply(lambda x: x.lower())
        
        df_3 = df7['boredom_signs'].value_counts().reset_index()
        df_3.columns = ['boredom_signs', 'count']
        
        
        def assign_value_3(emotion):
            if emotion == 'yawn':
                return -3
            elif emotion == 'sleep':
                return -7
        
        
        df_3['value'] = df_3['boredom_signs'].apply(assign_value_3)
        
        df_3['score'] = df_3['count'] * df_3['value'] * 0.2
        
        
        
        # Yawn_853_5January
        
        selectQuery = 'SELECT * FROM Yawn_853_5January'
        cursor = conn.execute(selectQuery)
        profile = 'None'
        l = []
        for row in cursor:
            l.append(row)
        
        df10= pd.DataFrame(l, columns=['emp_id', 'boredom_signs', 'timestamp'])
        
        df10 = df10.drop_duplicates()
        
        df10['time'] = df10['timestamp'].apply(lambda x: re.sub('\d{4}-\d{2}-\d{2} ', '', x))
        
        
        score_list = []
        
        score_list.append(df1['score'][0])
        score_list.append(df_4['score'][0])
        score_list = score_list + list(df_2['score']) + list(df_3['score'])
        
        
        final_sum = int(sum(score_list))
        
        if final_sum < 30:
            str1 = 'Looks like your teammate needs your attention!'
            
        elif final_sum > 30 and final_sum < 90:
            str1 = 'Your teammate seems to be fine.'
            
        elif final_sum > 90:
            str1 = 'Your teammate is excited to be in this workspace!'
            
        return final_sum, str1


    hi, str_hi = happy()
    
    conn = sqlite3.connect('EmployeeDatabase.db')
    selectQuery = 'SELECT * FROM Reaction_13745_5January'
    cursor = conn.execute(selectQuery)
    l = []
    for row in cursor:
        l.append(row)
    df = pd.DataFrame(l, columns=['emp_id', 'emotion', 'timestamp'])
    df['emotion'] = df['emotion'].apply(lambda x: x.strip())
    df['emotion'] = df['emotion'].apply(lambda x: x.lower())
    df2 = df['emotion'].value_counts().reset_index()
    df2.columns = ['emotion', 'count']

    plt.figure(figsize=(10, 10))
    sns.set(style="whitegrid")
    ax = sns.barplot(x="emotion", y="count", data=df2)
    fig = ax.get_figure()
    fig.savefig("static/plot1.png")

    conn.close()

    conn = sqlite3.connect('EmployeeDatabase.db')

    selectQuery = 'SELECT * FROM Reaction_13745_5January'
    cursor = conn.execute(selectQuery)
    l = []
    for row in cursor:
        l.append(row)

    df = pd.DataFrame(l, columns=['emp_id', 'emotion', 'timestamp'])
    df['emotion'] = df['emotion'].apply(lambda x: x.strip())
    df['emotion'] = df['emotion'].apply(lambda x: x.lower())

    df = df.drop_duplicates(subset=['timestamp'])

    df['time'] = df['timestamp'].apply(
        lambda x: re.sub('\d{4}-\d{2}-\d{2} ', '', x))

    def assign_value(emotion):
        if emotion == 'happy':
            return 10
        elif emotion == 'sad':
            return -8
        elif emotion == 'angry':
            return -10
        elif emotion == 'fear':
            return -5
        elif emotion == 'disgust':
            return -4
        elif emotion == 'surprise':
            return 5
        elif emotion == 'neutral':
            return 2

    df['value'] = df['emotion'].apply(assign_value)

    df['score'] = df['value'].cumsum()

    plt.figure(figsize=(15, 10))
    sns.set(style="whitegrid")
    ax = sns.lineplot(x="time", y="score", data=df)
    plt.xticks(rotation=90, horizontalalignment='right')
    fig = ax.get_figure()
    fig.savefig("static/plot2.png")

    conn.close()

    conn = sqlite3.connect('EmployeeDatabase.db')

    selectQuery = 'SELECT * FROM Sleep_13745_5January'
    cursor = conn.execute(selectQuery)
    l = []
    for row in cursor:
        l.append(row)

    df = pd.DataFrame(l, columns=['emp_id', 'boredom_signs', 'timestamp'])

    selectQuery2 = 'SELECT * FROM Yawn_13745_5January'
    cursor2 = conn.execute(selectQuery2)
    l2 = []
    for row in cursor2:
        l2.append(row)

    df2 = pd.DataFrame(l2, columns=['emp_id', 'boredom_signs', 'timestamp'])

    df = df.append(df2)

    df['boredom_signs'] = df['boredom_signs'].apply(lambda x: x.strip())
    df['boredom_signs'] = df['boredom_signs'].apply(lambda x: x.lower())

    #df = df.drop_duplicates()
    #df['time'] = df['timestamp'].apply(lambda x: re.sub('\d{4}-\d{2}-\d{2} ', '', x))

    df2 = df['boredom_signs'].value_counts().reset_index()
    df2.columns = ['boredom_signs', 'count']

    labels = 'Sleep', 'Yawn'
    sizes = df2['count'].tolist()
    explode = (0, 0.1)

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels,
            autopct='%1.1f%%', shadow=True, startangle=90)
    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.axis('equal')

    plt.savefig("static/plot3.png")

    conn.close()

    return render_template('dash.html', id=id, img_id='plot1', HI=hi, str_hi=str_hi)


@app.route("/dashboard/<id>/<img_id>/<hi>/<str_hi>")
def render_graph(id, img_id, hi, str_hi):
    return render_template('dash.html', id=id, img_id=img_id, HI=hi, str_hi=str_hi)


@app.route("/events")
def render_events():
    return render_template('events.html')


@app.route("/send_emails/<name>", methods=['POST'])
def send_da_mail(name):
     # Retrieve JSON inputs
    #response = name

    # Parse through JSON
    offer_name = name

    conn = sqlite3.connect('EmployeeDatabase.db')

    selectQuery = 'Select Email_Id from Employee'
    cursor = conn.execute(selectQuery)
    l = []
    for row in cursor:
        l.append(row)

    email_df = pd.DataFrame(l, columns=['email_id'])

    email_recipients = list(email_df['email_id'])

    coupon_code = ''.join(secrets.choice(
        string.ascii_uppercase + string.digits) for i in range(8))

    try:

        # Email Subject, Sender and Recipients
        msg = Message("Amazing Offer only for your eyes !",
                      sender="ops@ramco.com",
                      recipients=email_recipients)

        # Email Body
        msg.body = """Hello Ramcoite !\nYour hard work never goes unnoticed. Here is an amazing offer for you !\nSpend your time at - """ + \
            str(offer_name) + """ \n Use code - """ + str(coupon_code) + \
            """\nSent with regards,\n Your Friends @ Ramco"""

        # Send Email
        mail.send(msg)

        return redirect(url_for('emp'))

    except Exception as e:
        print('Error Encountered:', e, flush=True)
        return redirect(url_for('events'))


if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True, host='0.0.0.0', port=4000)
