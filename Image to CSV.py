import pandas as pd
import numpy as np
from matplotlib.image import imread
import os

base_dir = 'D:\\Fall 2022\\CV project\\dataset'
print(os.listdir(base_dir))

def image_to_csv(folder_name):
   
    single_image_dir = base_dir+"\\"+folder_name
    images = os.listdir(single_image_dir)
    li =[]
    for image in images:
        single_image = single_image_dir+"\\"+image
        img_arr = imread(single_image)
        reshape = img_arr.reshape((-1,))
        li.append(reshape)

    mild = np.array(li)

    sub_df = pd.DataFrame(mild)
    if folder_name == "Mild_Demented":
        sub_df['target'] = 0
    elif folder_name == "Moderate_Demented":
        sub_df['target'] = 1
    elif folder_name == "Non_Demented":
        sub_df['target'] = 2
    elif folder_name == "Very_Mild_Demented":
        sub_df['target'] = 3
    return sub_df

mild_df = image_to_csv("Mild_Demented")
moderate_df = image_to_csv("Moderate_Demented")
non_df = image_to_csv("Non_Demented")
very_df = image_to_csv("Very_Mild_Demented")
df = pd.concat([mild_df, moderate_df, non_df, very_df])
df.shape

print(df['target'].value_counts())

df.to_csv("ad_dataset.csv")

