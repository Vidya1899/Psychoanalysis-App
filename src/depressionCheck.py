# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 14:39:17 2020

@author: VIDYA
"""
# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import math
# construct the argument parse and parse the arguments
import csv

#from marker10 import *
import os, io
from google.cloud import vision
from google.cloud.vision_v1 import types
plotpath="/home/gvidya1899/Graphology-App/static/images/hist2.png"


contentDictionary = {
    "Normal":"Hey there ! Your handwriting is just as that of a normal person and doesn't show any sign  of dyslexia. It also indicates that you have strong focus and function to execute while writing.",
    "Depression":"Hey there ! Your handwriting indicates that you have a very high chance of having dyslexia. You might be finding it problematic to apply the right amount of pressure while writing"
}

def depression(file_name, image_path,sampl_no):
# load the image from disk
    image = cv2.imread(image_path)

# convert the image to grayscale and flip the foreground
# and background to ensure foreground is now "white" and
# the background is "black"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

# threshold the image, setting all foreground pixels to
# 255 and all background pixels to 0
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# grab the (x, y) coordinates of all pixel values that
# are greater than zero, then use these coordinates to
# compute a rotated bounding box that contains all
# coordinates
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

# the `cv2.minAreaRect` function returns values in the
# range [-90, 0); as the rectangle rotates clockwise the
# returned angle trends to 0 -- in this special case we
# need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)

# otherwise, just take the inverse of the angle to make
# it positive
    else:
        angle = -angle

# rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# draw the correction angle on the image so we can validate it
    cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# show the output image
    #print("angle: {:.3f}".format(angle))
    #cv2.imshow("Input", image)

    #cv2.imshow("Rotated", rotated)
    #cv2.waitKey(0)
    return(abs(angle))


        
def ocr1(file_name, image_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] ="/home/gvidya1899/.credentials/credentials.json"
    client = vision.ImageAnnotatorClient()

    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)
    response = client.document_text_detection(image=image)

    docText = response.full_text_annotation.text
    ocr=docText.rstrip('\n')
    return(ocr)
##############################################################################
#Func word passer converts ocr output to list of words 
depressed=["abandoned","achy","afraid","agitated","agony","alone","anguish","antisocial","anxious","breakdown","brittle","broken","catatonic","consumed","crisis","crushed","crying","defeated","defensive","dejected","demoralized","desolate","despair","desperate","despondent","devastated","discontented","disheartened","dismal","distractable","distraught","distressed","doomed","dreadful","dreary","edgy","emotional","empty","excluded","exhausted","exposed","fatalistic","forlorn","fragile","freaking","gloomy","grouchy","helpless","hopeless","hurt","inadequate","inconsolable","injured","insecure","irrational","irritable","isolated","lonely","lousy","low","melancholy","miserable","moody","morbid","needy","nervous","nightmarish","oppressed","overwhelmed","pain","paranoid","pessimistic","reckless","rejected","resigned","sadness","selfconscious","selfdisgust","shattered","sobbing","sorrowful","suffering","suicidal","tearful","touchy","trapped","uneasy","unhappy","unhinged","unpredictable","upset","vulnerable","wailing","weak","weepy","withdrawn","woeful","wounded","wretched"]
def wordpasser(ocr_text):
    word=ocr_text.replace(".","")
    #print(type(word))
    flag=0
    x=0
    for item in word.split():
        for words in depressed:
            if(words.lower()==item.lower()):
                flag+=1
                continue
            else:
                continue
    #print(x)
    #print(flag)
    return(flag)
#file_name = '44.png'


def plotgraph(trait_magnitude):
    # set font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica'

    # set the style of the axes and the text color
    plt.rcParams['axes.edgecolor']='#333F4B'
    plt.rcParams['axes.linewidth']=1
    plt.rcParams['xtick.color']='#333F4B'
    plt.rcParams['ytick.color']='#333F4B'
    plt.rcParams['text.color']='#333F4B'

    # create some fake data
    #percentages = pd.Series([94, 86, 18, 94,86,86, 10,90, 94, 92,86,92], 
    #                      index=['Marker 12', 'Marker 11', 'Marker 10', 'Marker 9', 
    #                            'Marker 8', 'Marker 7', 'Marker 6', 'Marker 5', 'Marker 4', 'Marker 3','Marker 2','Marker 1'])
    percentages = pd.Series(trait_magnitude, 
                            index=['Sadness','Depression','Suicidal','Frustration'])
    df = pd.DataFrame({'percentage' : percentages})
    #df = df.sort_values(by='percentage')

    # we first need a numeric placeholder for the y axis
    my_range=list(range(1,len(df.index)+1))

    fig, ax = plt.subplots(figsize=(5,1))

    # create for each expense type an horizontal line that starts at x = 0 with the length 
    # represented by the specific expense percentage value.
    plt.hlines(y=my_range, xmin=0, xmax=df['percentage'], color='#007ACC', alpha=0.2, linewidth=5)

    # create for each expense type a dot at the level of the expense percentage value
    plt.plot(df['percentage'], my_range, "o", markersize=5, color='#007ACC', alpha=0.6)

    # set labels
    ax.set_xlabel('Percentage', fontsize=15, fontweight='black', color = '#333F4B')
    ax.set_ylabel('')

    # set axis
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.yticks(my_range, df.index)

    # add an horizonal label for the y axis 
    fig.text(-0.01, 0.96, 'Trait', fontsize=15, fontweight='black', color = '#333F4B')
    #0.23
    # change the style of the axis spines
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)

    # set the spines position
    ax.spines['bottom'].set_position(('axes', -0.04))
    ax.spines['left'].set_position(('axes', 0.015))

    plt.savefig(plotpath, dpi=300, bbox_inches='tight')


def runInference(uniqueFilename):
    file_name= uniqueFilename #os.path.join("uploads/dyslexia/", uniqueFilename)
    sampl_no=1
    image_path = uniqueFilename #os.path.join("uploads/", uniqueFilename)
    skew=depression(file_name, image_path,sampl_no)

    ocr_text=ocr1(file_name,image_path)
    context_flag=wordpasser(ocr_text)

    trait_magnitude=[]
        
    if((skew>-0.2 or skew<0.2) and context_flag==0):
        trait_magnitude.append(20)
    elif((skew<-0.2 or skew>0.2) or context_flag>2):
        trait_magnitude.append(75)
    else:
        trait_magnitude.append(40)

    if((skew>-0.2 or skew<0.2) and context_flag==0):
        trait_magnitude.append(10)
    elif((skew<-0.2 or skew>0.2) or context_flag>2):
        trait_magnitude.append(60)
    else:
        trait_magnitude.append(50)

    if((skew>-0.2 or skew<0.2) and context_flag==0):
        trait_magnitude.append(2)
    elif((skew<-0.2 or skew>0.2) or context_flag>2):
        trait_magnitude.append(15)
    else:
        trait_magnitude.append(5)

    if((skew>-0.2 or skew<0.2) and context_flag==0):
        trait_magnitude.append(10)
    elif((skew<-0.2 or skew>0.2) or context_flag>2):
        trait_magnitude.append(50)
    else:
        trait_magnitude.append(25)

    plotgraph(trait_magnitude)
    if(skew>2 or context_flag>=3):
        return {"Heading":"Normal", "content":contentDictionary["Normal"]}
    elif(skew<=2 or context_flag<3):
        #return "Potential"
        return {"Heading":"Potential", "content":contentDictionary["Depression"]}
    else:
        return {"Heading":"INVALID SAMPLE", "content":"Sorry! Your sample couldn't be processed. Please try with a different sample."}
