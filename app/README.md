# BoobaBot, a rap music writer :pencil:
## The App

A very tiny and simple app using React and draft-js to power a very simple auto-complete editor for our A.I. backend

**Disclaimer:** *This app is not user friendly at all, it's more a proof of concept to use tensorflow in a server-like architecture*

## How to start the App
### Installing all depedencies
**First, make sure you ran the script to download the default RNN model weights**
```bash
cd app
npm install
cd front && npm install 
```
### Using the app with default parameters
We use foreman to launch the backend and the frontend together
```bash
npm start
```

### With custom parameters
You can also launch the back/front-end manually:
```bash
# First bash
python server.py --model_dir my/path/to/my/model

# Second bash
cd front and npm start
```

## Architecture
- **front:** Holds everything related to the app itself
- **current folder:** Holds everything related to the backend of the app, we were using Express and a very simple infamous `child_process.exec` :rage4: to call the python script but now we use a python server :bowtie:
- **files**:
  - **.env** and **Procile**: foreman files to launch default configuration app
  - **server_old.js**: This is a first implementation of a javascript Server using Express to call the python script `predict.py`. I rapidly moved away from this implementation as the simple idea of loading tensorflow and restoring weights for each call was disastrous in terms or performance.
  - **server.py**: A python server using Flask, where i create a Session once and for all, restore the weights and wait for an incoming request. We now have a decent latency on a CPU (~real-time for <5 words predictions).

Have fun! :beers: