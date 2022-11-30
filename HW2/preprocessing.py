import gensim.downloader as api

class DataSets:
    def __int__(self, paths_dict):
        self.paths_dict = paths_dict

    def create_datsets(self):
        for dataset_name, path in self.paths_dict():
            self.datasets_dict[dataset_name] = DataSets(path)


class DataSet:
    def __init__(self, path):
        self.embedder = None
        self.path = path
        is_tagged = 'tagged' in path
        all_sentences_x = []
        all_sentences_y = []
        sentence_x = []
        sentence_y = []
        with open(path) as f:
            for line in f.readlines():
                line = line[:-1] if line[-1] == '\n' else line
                word, tag = line.split('\t')
                if word == '':
                    assert tag == ''
                    all_sentences_x.append(sentence_x)
                    if is_tagged:
                        all_sentences_y.append(sentence_y)
                else:
                    if is_tagged:
                        sentence_x.append(word)
                        sentence_y.append(tag)
                    else:
                        sentence_x.append(word[0])
        self.X = all_sentences_x
        self.Y = all_sentences_y

    def embed_words(self, embedder):
        if embedder not in ['word2vec', 'glove']:
            print(f'{embedder} is not a familier embedder')
            raise NotImplemented
        if embedder == 'glove':
            self.embedder = api.load('glove-wiki-gigaword-50')
        elif embedder == 'word2vec':
            self.embedder = api.load('word2vec-ruscorpora-300')
            
        all_sentences_x_vectorized = []
        for sentence in self.X:
            all_sentences_x_vectorized.append

