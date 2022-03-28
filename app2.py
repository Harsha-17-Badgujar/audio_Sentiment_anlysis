# import re
# import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# STOPWORDS = set(stopwords.words('english'))

from unittest import result
from flask.templating import render_template_string
import requests
from flask import Flask,redirect,url_for,render_template,request,flash

import os
import librosa
import pandas as pd
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn import metrics
import glob


import pandas as pd
import numpy as np

import os
import sys

# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files we will see it later.
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# to play the audio files
from IPython.display import Audio

import pickle 


with open(r"C:/Users/harsha.badgujar/Downloads/archive (2)/knn_audio_model.pkl",'rb') as file:  
    Pickled_Model = pickle.load(file)
with open(r"C:/Users/harsha.badgujar/Downloads/archive (2)/labelonehot.pkl",'rb') as file:  
    onehot_encoding = pickle.load(file)



def speaker_diarization(audio_filename):
    
    def read_audio(audio_path):
        (sampling_rate, audio_data) = wavfile.read(audio_path)
        return sampling_rate, audio_data
    def visualize_audio_wave_over_time(sampling_rate, audio_data):
        time_array = np.arange(audio_data.size)/sampling_rate
    def get_audio_segments(audio_data, segment_size, sr):
        # Fix-sized segmentation (breaks a signal into non-overlapping segments)
        signal_len = len(audio_data)
        segment_size_t = segment_size # segment size in seconds
        segment_size = segment_size * sr  # segment size in samples
        # Break signal into list of segments in a single-line Python code
        segments = np.array([audio_data[x:x + segment_size] for x in np.arange(0, signal_len, segment_size)])
        return segments
    def normalize_audio(audio_data, sr):
        # Ref: https://hackernoon.com/audio-handling-basics-how-to-process-audio-files-using-python-cli-jo283u3y
        audio_data_norm = audio_data / (2**15)
        visualize_audio_wave_over_time(sr, audio_data_norm)
        return audio_data_norm   
    # Read preprocessed file
    audio_filename = audio_filename
    fs_wav, data_wav = read_audio(audio_filename)

    print('Signal Duration = {} seconds'.
        format(data_wav.shape[0] / fs_wav))

    print("Sampling rate: ", fs_wav)     

    data_wav_norm = normalize_audio(data_wav, fs_wav)

    segments = get_audio_segments(data_wav_norm, 1, fs_wav)
    len(segments)

    def get_mfcc_feature(ad, sr):
        mfccs = librosa.feature.mfcc(ad, sr=sr)
        return mfccs

    all_features = []
    for idx, arr in enumerate(segments):
        print(idx)
        
        features_list = []

        mfccs = get_mfcc_feature(arr, fs_wav)
        features_list.append(np.mean(librosa.feature.spectral_centroid(y=arr, sr=fs_wav)))
        features_list.append(np.mean(librosa.feature.spectral_bandwidth(y=arr, sr=fs_wav)))
        features_list.append(np.mean(librosa.feature.spectral_rolloff(y=arr, sr=fs_wav)))
        features_list.append(np.mean(librosa.feature.zero_crossing_rate(arr)))

        for mfcc in mfccs:
            features_list.append(np.mean(mfcc))
        
        all_features.append(features_list)
    df = pd.DataFrame(all_features)
    df.head()   

    # Feature scaling
    df_scaled = df.copy()

    for c in list(df.columns):
        df_column = df[c]
        min_value = df_column.min()
        max_value = df_column.max()
        norm_data = (df_column - min_value) / (max_value - min_value)
        df_scaled[c] = norm_data

    df_scaled.head()

    def simple_k_means(X, n_clusters=3, score_metric='euclidean'):
        model = KMeans(n_clusters=n_clusters)
        clusters = model.fit_transform(X)

        # There are many methods of deciding a score of a cluster model. Here is one example:
        score = metrics.silhouette_score(X, model.labels_, metric=score_metric)
        
        
        return dict(model=model, score=score, clusters=clusters)
    # Passing 5 clusters because there are 5 speakers - Change as per the video
    model_data = simple_k_means(df_scaled, n_clusters=3)
    model_data['model'].labels_
    df.to_csv("clusters/kmeans_clusters_without_noise_preprocessed_spectral_feats.csv", index=False)

    df['clusters'] = list(model_data['model'].labels_)

    #fig, ax = plt.subplots()
    #ax.hist(df['clusters'])
    #plt.show()
    #df   
    df.to_csv("clusters/kmeans_clusters_without_noise_preprocessed_spectral_feats.csv", index=False)
 
    def group_audios_based_on_clusters(dfx, audio_segments, sr):
        clusters = dfx['clusters']
        #print("clusters is:",clusters)
        label_indexes_dict = {}
        label_audio_clusters = {}

        for idx, label in enumerate(clusters):
            if label in label_indexes_dict.keys():
                label_indexes_dict[label].append(idx)
            else:
                label_indexes_dict[label] = [idx]

        for k, v in label_indexes_dict.items():
            # get segments that have energies higher than a the threshold:
            clustered_segments = audio_segments[v]

            # concatenate segments to signal:
            audio_cluster = np.concatenate(clustered_segments)

            label_audio_clusters[k] = audio_cluster

        return label_audio_clusters
    # Save all clusters
    def save_audio_clusters(dir_name, speaker_audio_dict, sr):
        if not os.path.exists("clusters/"+dir_name):
            os.makedirs("clusters/"+dir_name)

        for speaker, audio in speaker_audio_dict.items():
            wavfile.write("clusters/"+dir_name+"/"+str(speaker)+".wav", sr, audio)
    # Saving K-means audio clusters
    fs_wav, data_wav = read_audio(audio_filename)
    segments = get_audio_segments(data_wav, 1, fs_wav)

    df_kmeans_clusters_preprocessed = pd.read_csv("clusters/kmeans_clusters_without_noise_preprocessed_spectral_feats.csv")

    speaker_audio_dict = group_audios_based_on_clusters(df_kmeans_clusters_preprocessed, segments, fs_wav)
    save_audio_clusters("kmeans_clusters_without_noise_preprocessed_spectral_feats", speaker_audio_dict, fs_wav)

    file_list=[]
    duration_seconds_list=[]
    path = "E:/audio_Sentiment_anlysis/clusters/kmeans_clusters_without_noise_preprocessed_spectral_feats"
    files = os.listdir(path)
    for filename in glob.glob(os.path.join(path, '*.wav')):
        file_list.append(filename )
    for i in file_list:
    #print("i value is:",i)
        (source_rate, source_sig) = wavfile.read(i)
        duration_seconds = len(source_sig) / float(source_rate)
        duration_seconds_list.append(duration_seconds)    
        length_file = dict(zip(file_list, duration_seconds_list))
        Keymax = max(zip(length_file.values(), length_file.keys()))[1]
    print("selected file is:",Keymax)
    return Keymax     
def extract_features(data,file_name):
    data, sample_rate = librosa.load(file_name)
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result
def get_features1(path):
            # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    print("get1 data is:",data)        
            # without augmentation
            #result = extract_features1(data)
            #result = np.array(result)
            #result = np.vstack((result))
    res = extract_features(data,path)
    result = np.array(res)
    result = np.vstack((result,result))
            #print("result is:",result)
            
        #     # data with noise
        #     noise_data = noise(data)
        #     res2 = extract_features(noise_data)
        #     result = np.vstack((result, res2)) # stacking vertically
            
        #     # data with stretching and pitching
        #     new_data = stretch(data)
        #     data_stretch_pitch = pitch(new_data, sample_rate)
        #     res3 = extract_features(data_stretch_pitch)
            # stacking vertically
    print("finally feature extracted")    
    return result
def audio_prediction(data_file): 
    test_file_list=[]  
    test_file_list.append(data_file)
    feature_extract_test_file_list = []
    for path_test in (test_file_list):
        print("path_test is:",path_test)
        feature_test = get_features1(path_test)
        print("feature test array is:",feature_test)
        print("feature test array shape is:",feature_test.shape)
            #print("feature test is:",feature_test)
            #X1.append(feature_test)
        for ele1_test in feature_test:
            feature_extract_test_file_list.append(ele1_test)  
    Features_test_file_list = pd.DataFrame(feature_extract_test_file_list)
    Features_test_file_list.to_csv('features_test_file_list.csv', index=False)
    Features_test_file_list.head()   
    values_list = Features_test_file_list.iloc[:,:].values
    prediction_result = Pickled_Model.predict(values_list)
    final_result= onehot_encoding.inverse_transform(prediction_result)
    print("prediction is:",final_result) 
    return final_result 
     

app = Flask(__name__)
app.debug = True
# cache = Cache(app)
# app.config["CACHE_TYPE"] = "null"  
# cache.init_app(app)




@app.route('/')
def home():
    if request.method == 'POST':
        return render_template('info.html')
    return render_template('info.html')

@app.route('/info',methods=['GET','POST'])
def info():
    if request.method == 'POST':
        return render_template('info.html')
    return render_template('info.html')

@app.route('/login',methods=['GET','POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form["username"] != "harsha" or request.form["password"] != "harsha":
            error="Invalid Credentials. Please Try Again"
        elif request.form["username"] == "harsha" or request.form["password"] == "harsha":
            username = request.form['username']
            password = request.form['password']
            print("logged in")
            print("username password ",username, password)
            return redirect(url_for('info'))
    return render_template("login.html", error=error)

@app.route('/error',methods=['GET','POST'])
def error():
    if request.method == 'POST':
        return render_template('error.html')
    return render_template('error.html')

@app.route('/logout',methods=['GET','POST'])
def logout():
    if request.method == 'POST':
        print("logged out")
        return render_template('logout.html')
    return render_template('logout.html')

@app.route('/Audiouplod', methods=['GET','POST'])
def Audiouplod():
    if request.method == 'POST':
        uploaded_file=request.files['upload']  
        if uploaded_file :
            print("uploaded_file:",uploaded_file)
            
            max_file=speaker_diarization(uploaded_file)
            print("max file is:",max_file)
            audio_prediction_result=audio_prediction(max_file)
            print("audio_prediction_result:",audio_prediction_result)
        return render_template("Audiouplod.html",audio_prediction_result=audio_prediction_result[0])
        
    return render_template("Audiouplod.html")                                
        

        
if __name__ == '__main__':
    

    app.run(debug=True)
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    # app.config['SERVER_NAME'] = 'example.com'
    # with app.app_context():
    #      url_for('second_anysis', _external=True)