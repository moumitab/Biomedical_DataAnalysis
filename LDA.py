import numpy as np
import lda
import lda.datasets
import pandas as pd
data = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\conditions_clustering_analysis\\TopicModelling\\BagOfWord.csv'
vocab_data = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\conditions_clustering_analysis\\TopicModelling\\vocab.csv'

# X = lda.datasets.load_reuters()
# vocab = lda.datasets.load_reuters_vocab()
# titles = lda.datasets.load_reuters_titles()
# print(X.shape)
#
# print(X.sum())
# print(vocab)
# print len(vocab)
# model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
# model.fit(X)  # model.fit_transform(X) is also available
# topic_word = model.topic_word_  # model.components_ also works
# n_top_words = 8
# for i, topic_dist in enumerate(topic_word):
#     topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
#     print('Topic {}: {}'.format(i, ' '.join(topic_words)))



X = pd.read_csv(data)
X = np.asarray(X)
print(X.shape)
print(X.sum())
vocab = pd.read_csv(vocab_data)
vocab = list(vocab)
model = lda.LDA(n_topics=10, n_iter=1500, random_state=1)
model.fit(X)
topic_word = model.topic_word_

n_top_words = 8
print(model.ndz_)
print(model.nzw_)
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
