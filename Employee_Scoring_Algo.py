import sqlite3
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
conn = sqlite3.connect('EmployeeDatabase.db')


# EMPLOYEE

selectQuery = 'SELECT * FROM Employee'
cursor = conn.execute(selectQuery)
profile = 'None'
l = []
for row in cursor:
    l.append(row)


df1 = pd.DataFrame(l, columns=['emp_id', 'emp_name', 'emp_grade', 'emp_team', 'task_skills_req', 'emp_skills'])

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

df2 = df2.drop_duplicates(subset=['timestamp'])

df2['time'] = df2['timestamp'].apply(lambda x: re.sub('\d{4}-\d{2}-\d{2} ', '', x))

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


df7 = pd.DataFrame(l, columns=['emp_id', 'sleep', 'timestamp'])

df7['time'] = df7['timestamp'].apply(lambda x: re.sub('\d{4}-\d{2}-\d{2} ', '', x))


# Sleep_853_5January

selectQuery = 'SELECT * FROM Sleep_853_5January'
cursor = conn.execute(selectQuery)
profile = 'None'
l = []
for row in cursor:
    l.append(row)


df8 = pd.DataFrame(l, columns=['emp_id', 'sleep', 'timestamp'])

df8['time'] = df8['timestamp'].apply(lambda x: re.sub('\d{4}-\d{2}-\d{2} ', '', x))


# Yawn_13745_5January

selectQuery = 'SELECT * FROM Yawn_13745_5January'
cursor = conn.execute(selectQuery)
profile = 'None'
l = []
for row in cursor:
    l.append(row)

df9= pd.DataFrame(l, columns=['emp_id', 'yawn', 'timestamp'])

df9['time'] = df9['timestamp'].apply(lambda x: re.sub('\d{4}-\d{2}-\d{2} ', '', x))


# Yawn_853_5January

selectQuery = 'SELECT * FROM Yawn_853_5January'
cursor = conn.execute(selectQuery)
profile = 'None'
l = []
for row in cursor:
    l.append(row)

df10= pd.DataFrame(l, columns=['emp_id', 'yawn', 'timestamp'])

df10['time'] = df10['timestamp'].apply(lambda x: re.sub('\d{4}-\d{2}-\d{2} ', '', x))




# Holidays

selectQuery = 'Select * from Holidays'
cursor = conn.execute(selectQuery)
profile = 'None'
l = []
for row in cursor:
    l.append(row)

df11= pd.DataFrame(l, columns=['holiday_date'])

#df11['time'] = df9['timestamp'].apply(lambda x: re.sub('\d{4}-\d{2}-\d{2} ', '', x))


















