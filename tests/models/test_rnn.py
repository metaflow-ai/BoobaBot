import os, sys, unittest
import numpy as np

dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir + '/../..')

from models.rnn import RNN

word_to_id_dict = dict({
    'un': 0, 
    'test': 1,
    '.': 2
})
class TestRNN(unittest.TestCase):

    def test_build(self):
        config = {
            'restore_embedding': False,
            'vocab_size': 3,
            'embedding_size': 3,
            'word_to_id_dict': word_to_id_dict
        }
        rnn = RNN(config)
        self.assertEqual(True, True)    

if __name__ == "__main__":
    unittest.main()