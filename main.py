from sys import argv
from os import listdir, getcwd
from os.path import isfile, join
import subprocess
from pydub import AudioSegment 

def convert_to_8khz():
    for i in range(1, 6):
        path = getcwd() + '/voices/' + str(i)
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        for filename in onlyfiles:
            fullpathname = path + '/' + filename
            subprocess.check_output(['ffmpeg', '-i', fullpathname, '-ar', '8000', fullpathname, '-y'])

def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    trim_ms = 0 
    assert chunk_size > 0 
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size
    return trim_ms

def remove_silences():
    for i in range(1, 6):
        path = getcwd() + '/voices/' + str(i)
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        for filename in onlyfiles:
            fullpathname = path + '/' + filename
            sound = AudioSegment.from_file(fullpathname, format="wav")
            start_trim = detect_leading_silence(sound)
            end_trim = detect_leading_silence(sound.reverse())
            duration = len(sound)    
            trimmed_sound = sound[start_trim:duration-end_trim]
            trimmed_sound.export(fullpathname, format="wav")


def main():    
    if len(argv) < 2: command = ''
    else: command = argv[1]

    if command == 'preprocess':
        #convert_to_8khz()
        #remove_silences()

        # TODO convert into spectograms and divide data into train and test datasets
        pass
    elif command == 'train':
        #TODO add neural nets implementation
        pass
    elif command == 'test':
        #TODO add scoring implementation
        pass
    else:
        print('try running main script with one of the following commands: preprocess, train or test.')

if __name__ == '__main__':
    main()