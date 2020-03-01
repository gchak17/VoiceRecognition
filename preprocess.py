import warnings
warnings.filterwarnings("ignore")
from pydub import AudioSegment
from os import getcwd, listdir, makedirs
from os.path import isfile, join, exists
from matplotlib.pyplot import figure, Axes, NullLocator
from librosa import load as load_audio
from numpy import asanyarray, save, load, argmax
from keras.preprocessing.image import load_img, img_to_array
from keras.utils.np_utils import to_categorical

def wav_to_spectrogram(audio_path, save_path, spectrogram_dimensions=(64, 64), noverlap=16, cmap='gray_r'):
    data = load_audio(audio_path, mono=True)[0]
    fig = figure()
    fig.set_size_inches((spectrogram_dimensions[0]/fig.get_dpi(), spectrogram_dimensions[1]/fig.get_dpi()))
    ax = Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.specgram(data, cmap=cmap, Fs=2, noverlap=noverlap)
    ax.xaxis.set_major_locator(NullLocator())
    ax.yaxis.set_major_locator(NullLocator())
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0)

def dir_to_spectrogram(spectrogram_dimensions=(64, 64), noverlap=16, cmap='gray_r'):
    for i in range(1, 6):
        train_path = getcwd() + '/train_audio/' + str(i)
        train_spectogram_path = getcwd() + '/train_spectogram/' + str(i)
        if not exists(train_spectogram_path): makedirs(train_spectogram_path)
        train_wavs = [f for f in listdir(train_path) if isfile(join(train_path, f))]
        for train_wav in train_wavs:
            wav_to_spectrogram(train_path + '/' + train_wav, train_spectogram_path + '/' + train_wav.replace('.wav', '.png'),  
                spectrogram_dimensions=spectrogram_dimensions, noverlap=noverlap, cmap=cmap)
        
        test_path = getcwd() + '/test_audio/' + str(i)
        test_spectogram_path = getcwd() + '/test_spectogram/' + str(i)
        if not exists(test_spectogram_path): makedirs(test_spectogram_path)
        test_wavs = [f for f in listdir(test_path) if isfile(join(test_path, f))]
        for test_wav in test_wavs:
            wav_to_spectrogram(test_path + '/' + test_wav, test_spectogram_path + '/' + test_wav.replace('.wav', '.png'),  
                spectrogram_dimensions=spectrogram_dimensions, noverlap=noverlap, cmap=cmap)

def get_sets():
    train_set, test_set = [], [] 
    for i in range(1, 6):
        train_spectogram_path = getcwd() + '/train_spectogram/' + str(i)
        train_spectograms = [f for f in listdir(train_spectogram_path) if isfile(join(train_spectogram_path, f))]
        for train_spectogram in train_spectograms:
            img = load_img(train_spectogram_path + '/' + train_spectogram)
            train_set.append([img_to_array(img), i - 1])
        test_spectogram_path = getcwd() + '/test_spectogram/' + str(i)
        test_spectograms = [f for f in listdir(test_spectogram_path) if isfile(join(test_spectogram_path, f))]
        for test_spectogram in test_spectograms:
            img = load_img(test_spectogram_path + '/' + test_spectogram)
            test_set.append([img_to_array(img), i - 1])
    train_x = [item[0] for item in train_set]
    train_y = [item[1] for item in train_set]
    test_x = [item[0] for item in test_set]
    test_y = [item[1] for item in test_set]
    return train_x, train_y, test_x, test_y

def save_data_sets(train_x, train_y, test_x, test_y):
    train_x = asanyarray(train_x)
    train_y = asanyarray(train_y)

    test_x = asanyarray(test_x)
    test_y = asanyarray(test_y)
    
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    
    save('train_x', train_x)
    save('test_x', test_x)
    save('train_y', train_y)
    save('test_y', test_y)

def main():
    dir_to_spectrogram()
    train_x, train_y, test_x, test_y = get_sets()
    save_data_sets(train_x, train_y, test_x, test_y)

if __name__ == "__main__":
    main()