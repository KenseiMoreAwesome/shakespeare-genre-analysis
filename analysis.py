# prep the data
plays = {
    'Comedy': ['asyoulikeit', 'amidsummernightsdream', 'muchadoaboutnothing', 'twelthnight'],
    'Tragedy': ['macbeth', 'othello', 'kinglear', 'hamlet'],
    'History': ['henryivpart1', 'henryv', 'henryvipart2', 'richardii']
}

def read_files_into_string(filenames):
    strings = []
    for filename in filenames:
        with open(f'control_corpus_clean/{filename}.txt') as f:
            strings.append(f.read())
        return '\n'.join(strings)
    
plays_by_genre = {}
for genre, files in plays.items():
    plays_by_genre[genre] = read_files_into_string(files)

for genre in plays:
    print(plays_by_genre[genre][:100])