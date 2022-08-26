import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from tensorflow import keras



from typing import Union

from fastapi import FastAPI

def new_func():
  app = FastAPI()
  return app

app = new_func()





class ai:

  global ctkw
  global ctacs
  global ctst
  global ctty
  global ann
  

  data = pd.read_csv('clean_2.csv')
  score=[]
  for i in range(0, len(data)):
    if data['Stage'][i]=='misccases/purpose':
      score.append(round(np.random.uniform(4,5), 2))
    elif data['Stage'][i]=='plaintiff/petitionerevidence':
      score.append(round(np.random.uniform(9,10),2))  
    elif data['Stage'][i]=='misc/appearance':
      score.append(round(np.random.uniform(5,6), 2))
    elif data['Stage'][i]=='miscarguments':
      score.append(round(np.random.uniform(7,8),2))
    elif data['Stage'][i]=='trafficchallan':
      score.append(round(np.random.uniform(1,2),2))
    elif data['Stage'][i]=='trafficchalan':
      score.append(round(np.random.uniform(1, 2), 2))
    elif data['Stage'][i]=='issues':
      score.append(round(np.random.uniform(4,5),2))
    elif data['Stage'][i]=='prosecutionevidence':
      score.append(round(np.random.uniform(9, 10), 2))
    elif data['Stage'][i]=='charge':
      score.append(round(np.random.uniform(3,4),2))
    elif data['Stage'][i]=='arguments':
      score.append(round(np.random.uniform(7,8), 2))
    elif data['Stage'][i]=='misccases/appearance':
      score.append(round(np.random.uniform(5,6),2))
    elif data['Stage'][i]=='miscargument':
      score.append(round(np.random.uniform(7,8), 2))
    elif data['Stage'][i]=='misccase':
      score.append(round(np.random.uniform(6,7),2))
    elif data['Stage'][i]=='finalarguments':
      score.append(round(np.random.uniform(7,8),2))
    elif data['Stage'][i]=='misccases':
      score.append(round(np.random.uniform(6, 7), 2))
    elif data['Stage'][i]=='miscarguements':
      score.append(round(np.random.uniform(7,8),2))
    elif data['Stage'][i]=='order':
      score.append(round(np.random.uniform(8,9), 2))
    elif data['Stage'][i]=='defendant/respondentevidence':
      score.append(round(np.random.uniform(9,10), 2))
  
  del data['Unnamed: 0']
  

  dys = data['days']
  dys =np.array(dys)
  dys.shape= (len(dys), 1)
  dys = dys.astype(np.float64)

  
  x=data.values
  ctkw = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[0])], remainder='passthrough')
  tempkw = x[:, 1]
  tempkw.shape = (len(tempkw), 1)
  k1 = ctkw.fit_transform(tempkw)
  tempacs= x[:, 2]
  tempacs.shape=(len(tempacs),1 )
  ctacs = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[0])], remainder='passthrough')
  k2 = ctacs.fit_transform(tempacs)
  ctst = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[0])], remainder='passthrough')
  tempst = x[:, 3]
  tempst.shape= (len(tempst), 1)
  k3=ctst.fit_transform(tempst)
  ctty = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[0])], remainder='passthrough')
  tempty = x[:, 4]
  tempty.shape = (len(tempty), 1)
  k4=ctty.fit_transform(tempty)  
  final = hstack((k1,k2))
  final= hstack((final, k3))
  final = hstack((final, k4))
  final = hstack((final, dys))
  #final=np.column_stack((final, dys))
  
  final = final.toarray()
  
  score = np.array(score)
  score.shape = (len(score), 1)

    


  import tensorflow as tf
  ann = tf.keras.Sequential()
  ann.add(tf.keras.layers.Dense(units=100, activation='relu'))
  ann.add(tf.keras.layers.Dense(units=100, activation='relu'))
  ann.add(tf.keras.layers.Dense(units=100, activation='relu'))
  ann.compile(optimizer= 'adam' , loss='mean_squared_error' )

  ann.fit(final, score, batch_size=32, epochs=100)

  def hi(name):
    return ("heyyo")

  def fun(keywd, actsc, stge, cs, day):
    p1=ctkw.transform([[keywd]])
    p2= ctacs.transform([[actsc]])
    p3=ctst.transform([[stge]])
    p4=ctty.transform([[cs]])
    pre = hstack((p1,p2))
    pre= hstack((pre, p3))
    pre = hstack((pre, p4))
    pre = hstack((pre, [[day]]))
    ans = ann.predict([pre])
    return str(ans[0][0])
  
  
MODEL = ai
# filename = 'main.py'
# pickle.dump(MODEL, open(filename, 'wb'))
# loaded_model = pickle.load(open(filename, 'rb'))
  
# pickle.close()



@app.get("/")
def read_root():
  return {"ai": "score"}


@app.post("/aiscore")
def fetch_score(keywd: str, actsc: str, stge: str, cs: str, day: int):
  prediction = ai.fun(keywd, actsc, stge, cs, day)
  return{'prediction': prediction}
  

























