from sys import argv

def main():    
    if len(argv) < 2: command = ''
    else: command = argv[1]

    if command == 'preprocess':
        #TODO transform and divide data into train and test 
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