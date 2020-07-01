import random
from numba import njit
from numba.experimental import jitclass
from numba import float32, int64, int32
import numpy as np
from numpy.linalg import matrix_power

# Numba complains about using matrix_power
# TODO: Figure out if np.dot really would be faster
import warnings
warnings.filterwarnings('ignore')

@njit
def index2(array, item):
    for idx, val in enumerate(array):
        if val == item:
            return idx

spec = [
    ('trans', float32[:,:]),
    ('count', int64[:,:]),
    ('limit', float32[:]),
    ('last_state', int32),
    ('nstates', int32),
]


@jitclass(spec)
class MarkovModel:
    def __init__(self, N: np.int32):
        self.trans  = np.zeros((N,N), dtype=np.float32)
        self.count = np.zeros((N,N), dtype=np.int64)
        self.limit = np.zeros(N, dtype=np.float32)
        self.last_state = -1
        self.nstates = N

    def add(self, state: np.int32):
        if self.last_state == -1:
            self.last_state = state
            return
        self.count[self.last_state, state] += 1
        self.last_state = state

    def reset(self):
        self.trans  = np.zeros((N,N))
        self.count = np.zeros((N,N))
        self.last_state = -1

    def update_transition_matrix(self):
        for i in range(self.nstates):
            self.trans[i,:] = np.cumsum(self.count[i,:]/self.count[i,:].sum())

        # This is a one-liner for the above loop, but the axis keyword to cumsum
        # isn't supported by numba
        #self.trans = np.cumsum((self.count.T/self.count.sum(axis=1)).T, axis=1).astype(np.float32)
        self.limit = np.cumsum(matrix_power(self.trans, self.nstates)[0,:])
        self.limit[nstates-1] = 1.

    def get(self):
        r = random.random()
        #print(r)
        if self.last_state == -1:
            self.last_state = index2(r < self.limit, True)
            return self.last_state
        else:
            self.last_state = index2(r < self.trans[self.last_state,:], True)
            return self.last_state

# From user MSeifert on StackOverflow
# https://stackoverflow.com/a/41578614
@njit
def index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx

states = {0:'WH', 1:'WM', 2:'RH', 3:'RM'}
nstates = len(states)
prob = np.cumsum([.05, .1, .7, .15])
#prob = np.cumsum([.25, .25, .25, .25])
seq_len = 10000
random.seed(1337)

sequence = []

for i in range(seq_len):
    sequence.append(index(random.random() < prob, True)[0])


##print(index2([False, False, False, True, True], True))
#exit()
MM = MarkovModel(4)

for s in sequence:
    MM.add(s)
MM.update_transition_matrix()

print(MM.trans)
print(sum(MM.limit))

pred = []

for i in range(1000):
    pred.append(MM.get())

for i in range(4):
    print(pred.count(i) / len(pred))

