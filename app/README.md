# BoobaBot, a rap music writer :pencil:
## The App

A very tiny and simple app using React and draft-js to power a very simple auto-complete editor for our A.I. backend

## How to start the App
Simple:
```bash
cd app
npm install
cd front && npm install && cd ..
npm start
```

## Architecture
- **front:** Holds everything related to the app itself
- **current folder:** Holds everything related to the backend of the app, we are using Express and a very simple infamous `child_process.exec` :rage4: to call the python script.

Have fun! :beers: