import os, sys, unittest
import numpy as np

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
        self.assertEqual(len(corpus), 7)

    def test_get_corpus_with_paragraph(self):
        corpus = util.clean_textfile(para_fixture_filepath)
        corpus_para = util.get_corpus_with_paragraph(corpus)
        self.assertEqual(len(corpus_para), 2)

    def test_word_to_id(self):
        wti_dict = {
            'test':0,
            '<UNK>':1
        }
        self.assertEqual(util.word_to_id(wti_dict, 'test'), 0)        
        self.assertEqual(util.word_to_id(wti_dict, 'other'), 1)  

if __name__ == '__main__':
    unittest.main()