import os, sys, unittest
import numpy as np
from collections import Counter

dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir + '/../..')

import util

para_fixture_filepath = dir + '/fixture/para.txt'

class TestUtil(unittest.TestCase):

    def test_clean_text(self):
        text = 'æœìíîïýÿòóôõöáâãäëñûüx2,X2'
        cleaned_text = util.clean_text(text)    

        true_text = ['aeoeiiiiyyoooooaaaaenuu', ',']
        self.assertEqual(cleaned_text, true_text)

    def test_clean_textfile(self):
        corpus = util.clean_textfile(para_fixture_filepath)
        counter = Counter()
        counter.update(corpus)
        self.assertEqual(len(corpus), 38)
        self.assertEqual(counter['<EOL>'], 7)
        self.assertEqual(counter['<EOP>'], 3)

    def test_word_to_id(self):
        wti_dict = {
            'test':0,
            '<UNK>':1
        }
        self.assertEqual(util.word_to_id(wti_dict, 'test'), 0)        
        self.assertEqual(util.word_to_id(wti_dict, 'other'), 1)  

    def test_makte_sets(self):
        corpus = util.clean_textfile(para_fixture_filepath)
        wti_dict = {word: i for i, word in enumerate(set(corpus))}

        train_set, dev_set, test_set = util.make_sets(corpus, wti_dict, .34)

        self.assertEqual(len(train_set), 13)        
        self.assertEqual(len(dev_set), 18)        
        self.assertEqual(len(test_set), 7)        
    

if __name__ == '__main__':
    unittest.main()