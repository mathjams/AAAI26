import pandas as pd
import os
def plainsequences(user_type):
  resulteye=[]
  resulthand=[]
  maxlen=0
  eye_basic_url='/Users/emilyyu/Downloads/Archive_4/AAAI/data_set/Eye_'
  hand_basic_url='/Users/emilyyu/Downloads/Archive_4/AAAI/data_set/Hand_'
  if (user_type=='ASD'):
    numOfUser=9
    eye_basic_url+="ASD_"
    hand_basic_url+='ASD_'
  else:
    eye_basic_url+="TD_"
    hand_basic_url+='TD_'
    numOfUser=17
  for i in range(1, numOfUser+1):
    for j in range(0,3):
      c_eye_url=eye_basic_url+'U'+str(i)+"_Active_"+str(j)+".xlsx"
      c_hand_url=hand_basic_url+'U'+str(i)+"_Active_"+str(j)+".xlsx"
      #asd_eye_data=pd.DataFrame()
      try:
        asd_eye_data=pd.read_excel(c_eye_url)
        asd_hand_data=pd.read_excel(c_hand_url)
        resulteye.append(asd_eye_data[['x','y']].to_numpy())
        resulthand.append(asd_hand_data[['x','y']].to_numpy())
        c_max_length=max(asd_eye_data.shape[0], asd_hand_data.shape[0])
        if c_max_length>maxlen:
          maxlen=c_max_length
      except IOError:
        print("")
  return resulteye, resulthand, maxlen