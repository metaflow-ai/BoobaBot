# BoobaBot, a rap music writer :pencil:
Hello! Welcome to the BoobaBot repository. 

This project is about developping an A.I. rap writer that can write a word/sentence/para/full song or autocomplete your writing to help you find some inspiration!

For more techy people: this is a basic GloVe embedding using LSTM best practices and TensorFlow to predict words.

You can also train your own bot on a different corpus to potentially create whatever artiste you'd like (rock and roll, law, poems, etc.)

**Disclaimer:** *The weights are not provided with the model*

## Installation
**Install depedencies**
```bash
pip install -r requirements.txt
```

## How to train it
You can train your GloVe on a corpus using the `train_glove.py` script with those options:
- **debug**: Debug mode
- **random_search**: Procede to a random search for hyperparameters tuning
- **textfile**: The textfile to use to build the corpus
- **embedding_size**: Embedding size
- **context_size**: Number of words to use for context
- **num_epochs**: How many epochs should we train the GloVe
- **nb_search_iter**: Batch size
- **stem**: Should we stem words?

You can train your model on a corpus using the `train.py` script with those options:
- **debug**: Debug mode
- **profiling**: Profiling mode, output a timeline readable inside chrome
- **num_epochs**: How many epochs should we train the RNN
- **batch_size**: Batch size
- **lr**: Learning rate for the adam optimizers
- **textfile**: The textfile used to train GloVe
- **glove_dir** The GloVe directory
- **train_glove**: Are we finetuning/training GloVe embedding
- **cell_name**: str, help="Cell architecture
- **rnn_activation**: str, help="RNN activation function names)")
- **seq_length**: RNN sequence length
- **state_size**: RNN state size
- **num_layers**: How deep is the RNN
- **tye_embedding**: Tye input/output word embedding weights 

## Let's make him rap baby!
Before make him rap, you need to train the model (see previous point).

You can make your model rap using the `predict.py` script with those options:
- **debug**: Debug mode 
- **model_dir**: The trained RNN dir
- **inputs**: Choose the beginning of the predicted sentence (see this as an initialization)
- **random**: Add some randomness, predict using temperature
- **temperature**: The temperature for predictions
- **top_k**: Return the top K prediction
- **nb_word**: How many words should it return
- **nb_sentence**: How many lines should it return
- **nb_para**: How many paragraph should it return

## launch the app:
**See the [README.md file](app/README.md) in the app folder **

## Running tests
```bash
python -m unittest discover
```

## Architecture
- **app:** Holds everything related to the autocomplete app, for more information on this point, check the README file inside the folder
- **crawler:** This is a very simple and rough crawler, but still, it made the work for us, so if you want to use it, go on!
- **dataviz:** Holds all python script to visualize the word embedding using sklearn or tensorboard
- **models:** Holds models python files: RNNs and GloVe
- **server:** Holds some files to use TensorFlow serving architecture in production
- **tests:** Holds some tests sherlocks!
- Python files:
  - **hyperparams_search.py:** Bash script to handle hyperparamerters search on the model 
  - **predict.py:** Generate new image from a trained model
  - **train_glove.py:** Train GloVe embedding
  - **train.py:** Train a model to achieve deep rap
  - **util.py:** Some utilitary function to clean and handles input data

Have fun! :beers:

# Reference
The GloVe embedding script has been taken from [this repository](https://github.com/GradySimon/tensorflow-glove) (Thank You! :thumbsup:) and be adapted to be more TensorFlow friendly (especially the saving/loading part)