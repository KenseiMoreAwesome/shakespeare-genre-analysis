import nltk
nltk.download('punkt')

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
    ## Building joint corpus
    joint_corpus = (plays_by_genre_tokens[genre] + plays_by_genre_tokens['Testing'])
    join_freq_dist = nltk.FreqDist(joint_corpus)
    most_common = list(join_freq_dist.most_common(500))

    genre_share = (len(plays_by_genre_tokens[genre]) / len(joint_corpus))

    ## Determines the frquency of certain words in genre compared to what would be expected
    chisquared = 0
    for word,joint_count in most_common:
        genre_count = plays_by_genre_tokens[genre].count(word)
        testing_count = plays_by_genre_tokens['Testing'].count(word)

        expected_genre_count = joint_count * genre_share
        expected_testing_count = joint_count * (1-genre_share)

    chisquared += ((genre_count - expected_genre_count) * (genre_count - expected_genre_count) / expected_genre_count)
    chisquared += ((testing_count - expected_testing_count) * (testing_count - expected_testing_count) / expected_testing_count )

    print(f'The Chisquared stat for genre {genre} is {chisquared}')

