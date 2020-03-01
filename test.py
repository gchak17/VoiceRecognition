import warnings
warnings.filterwarnings('ignore')

from keras.models import Sequential, load_model
from xlwt import Workbook
from numpy import asanyarray, save, load, argmax, round
from os import getcwd, listdir, makedirs
from os.path import isfile, join, exists
from keras.preprocessing.image import load_img, img_to_array
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav

def convert_to_spectogram(src_path, dest_path):
    spectrogram_dimensions = (64, 64)
    noverlap = 16
    cmap = 'gray_r'
    _, samples = wav.read(src_path)
    if len(samples.shape) != 1: samples = samples[:, 0]
    fig = plt.figure()
    fig.set_size_inches((spectrogram_dimensions[0] / fig.get_dpi(), spectrogram_dimensions[1] / fig.get_dpi()))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.specgram(samples, cmap=cmap, Fs=2, noverlap=noverlap)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    fig.savefig(dest_path, bbox_inches='tight', pad_inches=0)
    return dest_path

def get_spectograms():
    res = []
    voice_dir = 'test_audios/'
    file_names = [f for f in listdir(voice_dir) if isfile(join(voice_dir, f)) and f.endswith('.wav')]
    for file_name in file_names:
        src_path = voice_dir + file_name
        res.append(convert_to_spectogram(src_path, 'test_spectograms/' + file_name[:-3] + 'png'))
    return res

def get_test_set(spectograms):
    test_set = []
    for _, spectogram in enumerate(spectograms): test_set.append(img_to_array(load_img(spectogram)))
    x_test = asanyarray(test_set)
    x_test /= 255
    return x_test, spectograms

def main():
    model = load_model('model.h5')
    x_test, spectograms = get_test_set(get_spectograms())
    #correct_sum, all_sum = 0.0, 0.0#
    wb = Workbook()
    sh = wb.add_sheet('predictions')
    for index, data in enumerate(model.predict(x_test)):
        name = ((spectograms[index].split('/'))[-1]).replace('png', 'wav')
        #correct_sum += data[int(name[0])]#
        #all_sum += 1#
        sh.write(index, 0, name)
        for i in range(1, 6): sh.write(index, i, str(round(data[i], 2)))
    #print(correct_sum/all_sum)#
    wb.save('group_4.xls')

if __name__ == '__main__':
    main()