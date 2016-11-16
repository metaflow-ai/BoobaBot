import os, sys, unittest, shutil

dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir + '/../..')

import train_glove

para_fixture_filepath = dir + '/fixture/para.txt'

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class TestUtil(unittest.TestCase):

    def test_main(self):
        args = Namespace(
            debug=False,
            random_search=False,
            textfile=dir + '/fixture/lorem_ipsum.txt',
            embedding_size=3,
            context_size=4,
            num_epochs=2, 
            nb_search_iter=2,
            stem=True
        )
        log_dirs = train_glove.main(args)
        self.assertEqual(len(log_dirs), 2)

        for log_dir in log_dirs:
            shutil.rmtree(log_dir)


if __name__ == '__main__':
    unittest.main()