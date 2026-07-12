import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# prep the data
plays = {
    'Comedy': ['asyoulikeit', 'amidsummernightsdream', 'muchadoaboutnothing', 'twelthnight'],
    'Tragedy': ['macbeth', 'othello', 'kinglear', 'hamlet'],
    'History': ['henryivpart1', 'henryv', 'henryvipart2', 'richardii'],
    'Testing': ['macbeth', 'othello', 'kinglear', 'hamlet', 'asyoulikeit', 'amidsummernightsdream', 'muchadoaboutnothing', 'twelthnight', 'henryivpart1', 'henryv', 'henryvipart2', 'richardii']
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
    print(f"{genre}: {plays_by_genre[genre][:100]!r}")

# Tokenize all genres, including Testing
plays_by_genre_tokens = {}
for genre, text in plays_by_genre.items():
    plays_by_genre_tokens[genre] = [token.lower() for token in nltk.word_tokenize(text)]

# Kilgarriff’s Chi-Squared Method
genres = ('Comedy', 'History', 'Tragedy')

for genre in genres:
    joint_corpus = plays_by_genre_tokens[genre] + plays_by_genre_tokens['Testing']
    joint_freq_dist = nltk.FreqDist(joint_corpus)
    most_common = list(joint_freq_dist.most_common(500))

    genre_share = len(plays_by_genre_tokens[genre]) / len(joint_corpus)

    chisquared = 0.0
    for word, joint_count in most_common:
        genre_count = plays_by_genre_tokens[genre].count(word)
        testing_count = plays_by_genre_tokens['Testing'].count(word)

        expected_genre_count = joint_count * genre_share
        expected_testing_count = joint_count * (1 - genre_share)

        if expected_genre_count > 0:
            chisquared += ((genre_count - expected_genre_count) ** 2) / expected_genre_count
        if expected_testing_count > 0:
            chisquared += ((testing_count - expected_testing_count) ** 2) / expected_testing_count

    print(f'The Chi-squared stat for genre {genre} is {chisquared:.3f}')

