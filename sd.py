import os
import librosa
import pandas as pd
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn import metrics

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
            
if __name__ == '__main__':
    speaker_diarization("5.wav")        

        




