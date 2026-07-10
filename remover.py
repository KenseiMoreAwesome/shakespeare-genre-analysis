import os
import re
import shutil

def remove_direction(inputFolder, outputFolder):
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
        print(f"Created output folder: {outputFolder}")

    pattern = r'\[.*?\]'

    try:
        files = os.listdir(inputFolder)
    except FileNotFoundError:
        print(f"Input folder '{inputFolder}' does not exist.")
        return
    
    count = 0
    for filename in files:
        if filename.endswith('.txt'):
            inputFilePath = os.path.join(inputFolder, filename)
            outputFilePath = os.path.join(outputFolder, filename)

            try:
                with open(inputFilePath, 'r', encoding='utf-8') as f:
                    content = f.read()

                cleanContent = re.sub(pattern, '', content)

                with open(outputFilePath, 'w', encoding='utf-8') as f:
                    f.write(cleanContent)

                count += 1
                print(f"Processed file: {filename}")

            except Exception as e:
                print(f"Error processing file '{filename}': {e}")

    print(f"Total files processed: {count}")
    print(f"Cleaned files saved in: {outputFolder}")

if __name__ == "__main__":
    inputFolder = "control_corpus"
    outputFolder = "control_corpus_clean"
    
    remove_direction(inputFolder, outputFolder)