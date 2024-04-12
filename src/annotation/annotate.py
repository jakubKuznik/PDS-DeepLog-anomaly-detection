# Faculty: BUT FIT 
# Course: PDS
# Project Name: Log Analysis using ML.
# Name: Jakub Kuznik
# Login: xkuzni04
# Year: 2024

# Annotate data and put to stdout

import sys

def load_files_to_dict(filename1):
    data_dict = {}
    with open(filename1, 'r') as file1:
        for line1 in file1:
            key1, value1 = line1.strip().split(',')
            data_dict[key1] = value1
    return data_dict

def annotate_logs(file, data_dict):
    with open(file, 'r') as file1:
        for line1 in file1:
            words = line1.strip().split()
            for w in words:
                if w.startswith('blk_'):
                    block_num = w.split('_')
                    anot = data_dict[w]
                    new_word = block_num[0] + "_" + anot
                    w = new_word
                print(w, end=' ')    
            print()    

def main():
    print(sys.argv)
    if len(sys.argv) < 3:
        print("Usage: python program.py <file-annotations> <file-logs>")
        sys.exit(1)

    file_anot = sys.argv[1]
    file_logs = sys.argv[2]
    
    data_dict = load_files_to_dict(file_anot)
    annotate_logs(file_logs, data_dict)



if __name__ == "__main__":
    main()

