import pickle
from itertools import chain
import numpy as np

data = pickle.load(open('datasets/diginetica/all_train_seq.txt', 'rb'))
data = list(chain.from_iterable(data))
node = np.unique(np.asarray(data))

print('//')