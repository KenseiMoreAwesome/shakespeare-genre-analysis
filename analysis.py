import nltk
nltk.download('punkt')
%matplotlib inline

# prep the data
plays = {
    'Comedy': ['asyoulikeit', 'amidsummernightsdream', 'muchadoaboutnothing', 'twelthnight'],
    'Tragedy': ['macbeth', 'othello', 'kinglear', 'hamlet'],
    'History': ['henryivpart1', 'henryv', 'henryvipart2', 'richardii'],
    'Testing': []
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

#Tokenise
genres = ('Comedy', 'History', 'Tragedy')
plays_by_genre_tokens = {}
for genre in genres:
    tokens = nltk.word_tokenize(plays_by_genre[genre])

# Kilgariff’s Chi-Squared Method
#Lowercasing
for genre in genre:
    plays_by_genre_tokens[genre] = ([token.lower() for token in plays_by_genre_tokens[genre]])
plays_by_genre_tokens['Testing'] = ([token.lower() for token in plays_by_genre_tokens['Testing']])

#Calculating Chi-Squared value
for genre in genres:
    joint_corpus = ()