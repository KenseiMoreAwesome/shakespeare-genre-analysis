import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# Define the reference genres and the target test corpus.
plays = {
    'Comedy': ['asyoulikeit', 'amidsummernightsdream', 'muchadoaboutnothing', 'twelthnight'],
    'Tragedy': ['macbeth', 'othello', 'kinglear', 'hamlet'],
    'History': ['henryivpart1', 'henryv', 'henryvipart2', 'richardii'],
    'Testing': ['macbeth']
}

def read_files_into_string(filenames):
    # Returns combined text from corpus.
    strings = []
    for filename in filenames:
        try:
            with open(f'confirmed_corpus_clean/{filename}.txt', encoding='utf-8') as f:
                strings.append(f.read())
        except FileNotFoundError:
            try:
                with open(f'test_corpus_clean/{filename}.txt', encoding='utf-8') as f:
                    strings.append(f.read())
            except FileNotFoundError:
                print(f"File not found: {filename}.txt")
    return '\n'.join(strings)


def build_genre_corpora(genre_map):
    # Combines whole genre into one corpus
    corpora = {}
    for genre, files in genre_map.items():
        corpora[genre] = read_files_into_string(files)
    return corpora


def print_corpus_previews(corpora):
    #Print preview 
    for genre, text in corpora.items():
        print(f"{genre}: {text[:100]!r}")


def tokenize_corpora(corpora):
    #Tokenise and lowercase each corpus
    return {
        genre: [token.lower() for token in nltk.word_tokenize(text)]
        for genre, text in corpora.items()
    }


def compute_chi_squared(reference_tokens, test_tokens, top_n=500):
    joint_corpus = reference_tokens + test_tokens
    joint_freq_dist = nltk.FreqDist(joint_corpus)
    most_common = list(joint_freq_dist.most_common(top_n))

    reference_share = len(reference_tokens) / len(joint_corpus)
    chisquared = 0.0

    for word, joint_count in most_common:
        genre_count = reference_tokens.count(word)
        testing_count = test_tokens.count(word)

        expected_genre_count = joint_count * reference_share
        expected_testing_count = joint_count * (1 - reference_share)

        if expected_genre_count > 0:
            chisquared += ((genre_count - expected_genre_count) ** 2) / expected_genre_count
        if expected_testing_count > 0:
            chisquared += ((testing_count - expected_testing_count) ** 2) / expected_testing_count

    return chisquared


def main():
    plays_by_genre = build_genre_corpora(plays)
    print_corpus_previews(plays_by_genre)

    plays_by_genre_tokens = tokenize_corpora(plays_by_genre)

    genres = ('Comedy', 'Tragedy', 'History',)

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

        print('Testing')
        print(f'The Chi-squared stat for genre {genre} is {chisquared:.3f}')


if __name__ == '__main__':
    main()

