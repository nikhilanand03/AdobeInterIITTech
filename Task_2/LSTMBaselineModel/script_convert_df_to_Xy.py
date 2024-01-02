import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import ast


def getXYtrain(whichset):
    if(whichset=="train"):
        df = pd.read_csv("task2/img_video_embed.csv")[:51438]
        df_content_embed=pd.read_csv("task2/train_embeddings.csv")
    

    merged_df = pd.merge(df, df_content_embed, on='id')

    label_encoder = LabelEncoder()
    merged_df['username_encoded'] = label_encoder.fit_transform(merged_df['username_x'])
    merged_df['company_encoded'] = label_encoder.fit_transform(merged_df['inferred company_x'])

    reference_date = pd.to_datetime('2018-01-01')
    merged_df['date_feature'] = (pd.to_datetime(merged_df['date_x']) - reference_date).dt.days

    merged_df.rename(columns={'likes_x': 'likes','embedding':'content_embedding'}, inplace=True)

    merged_df['username_encoded'] = merged_df['username_encoded']/np.max(merged_df['username_encoded'])
    merged_df['company_encoded'] = merged_df['company_encoded']/np.max(merged_df['company_encoded'])
    merged_df['date_feature'] = merged_df['date_feature']/np.max(merged_df['date_feature'])

    X = []
    print("Making X")
    for i in range(len(merged_df)):
        row = merged_df.iloc[i]
        s_img = row['image_embed']
        s_vid = row['video_embed']
        user = row['username_encoded']
        comp = row['company_encoded']
        date = row['date_feature']

        if(isinstance(s_img,str)):
            imgvid = 0 
            arr_img=np.array(ast.literal_eval(s_img)).squeeze()
            arr_vid=np.zeros((768,))
        elif(isinstance(s_vid,str)):
            imgvid=1
            arr_img=np.zeros((512,))
            arr_vid=np.array(ast.literal_eval(s_vid)).squeeze()

        X.append(np.concatenate((arr_img, arr_vid, np.array([user,comp,date]))))
        if(i%1000==1):
            print(i)

    # X_train = np.array(X)
    # X_train.shape

    y = []
    print("Making y")
    for i in range(len(merged_df)):
        row = merged_df.iloc[i]
        content = row['content_embedding']

        cleaned_string = merged_df['content_embedding'][1].replace('[', '').replace(']', '').replace('\n', '').split()
        converted_values = [float(value) for value in cleaned_string]
        arr = np.array(converted_values).reshape(4, -1).flatten()
        y.append(arr)

        if(i%1000==1):
            print(i)

    return np.array(X),np.array(y)