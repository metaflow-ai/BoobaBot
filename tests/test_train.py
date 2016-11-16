import os, sys, unittest, shutil
from collections import Counter

dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir + '/../..')

import train, train_glove

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
textfile = dir + '/fixture/lorem_ipsum.txt'

class TestUtil(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = Namespace(
            debug=False,
            random_search=False,
            textfile=textfile,
            embedding_size=3,
            context_size=4,
            num_epochs=2, 
            nb_search_iter=1,
            stem=True
        )
        log_dirs = train_glove.main(args)
        cls.tmp_glove_dir = log_dirs[0]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_glove_dir)

    def test_main(self):
        args = Namespace(
            debug=False,
            profiling=False,
            num_epochs=2, 
            batch_size=2,
            lr=1e-3, 

            textfile= textfile,
            glove_dir=self.tmp_glove_dir,
            train_glove=True,
            
            cell_name='lstm',
            rnn_activation='tanh',
            seq_length=4,
            state_size=4,
            num_layers=1,
            tye_embedding=False
        )
        log_dir = train.main(args)
        shutil.rmtree(log_dir)

    def test_main_tye_embedding(self):
        args = Namespace(
            debug=False,
            profiling=False,
            num_epochs=2, 
            batch_size=2,
            lr=1e-3, 

            textfile= textfile,
            glove_dir=self.tmp_glove_dir,
            train_glove=True,
            
            cell_name='lstm',
            rnn_activation='tanh',
            seq_length=4,
            state_size=4,
            num_layers=1,
            tye_embedding=True
        )
        log_dir = train.main(args)
        shutil.rmtree(log_dir)

    def test_main_multiple_layers(self):
        args = Namespace(
            debug=False,
            profiling=False,
            num_epochs=2, 
            batch_size=2,
            lr=1e-3, 

            textfile= textfile,
            glove_dir=self.tmp_glove_dir,
            train_glove=True,
            
            cell_name='lstm',
            rnn_activation='tanh',
            seq_length=4,
            state_size=4,
            num_layers=3,
            tye_embedding=False
        )
        log_dir = train.main(args)
        shutil.rmtree(log_dir)


if __name__ == '__main__':
    unittest.main()