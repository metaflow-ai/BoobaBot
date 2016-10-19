import React, { Component } from 'react';
import { Editor, EditorState, Modifier } from 'draft-js';
import update from 'react-addons-update';

import request from 'superagent';

import './App.css';
import '../../node_modules/draft-js/dist/Draft.css'


class App extends Component {

  constructor(props) {
    super(props)

    this.state = {
      editorState: EditorState.createEmpty(),
      form: {
        number: 1,
        kind: 'words',
        temperature: 1,
        random: false,
        topk: 1,
      },
      loading: false,
      error: false,
      errorMessage: ""
    }

    this.focus = () => this.refs.editor.focus()
    this.onChange = this.onChange.bind(this)
    this.onFormChange = this.onFormChange.bind(this)
    this.submitForm = this.submitForm.bind(this)
    this.addEditorText = this.addEditorText.bind(this)
  }


  onChange(editorState) {
    this.setState({editorState})
  }


  onFormChange(e) {
    var name = e.target.name
    var val = e.target.value

    var newState = update(this.state, {form: {[name]: {$set: val}}})
    this.setState(newState)
  }


  submitForm(e) {
    const { editorState } = this.state
    const content = editorState.getCurrentContent()
    var inputs = content.getPlainText()

    inputs = inputs.replace(/\n{2,}/g, " <EOP> ")
    inputs = inputs.replace(/\n{1}/g, " <EOL> ")

    const url = "http://127.0.0.1:3001/api/predict"

    var paragraphs = []
    var lines = []
    var finalText = ""

    var args = update(this.state.form, {inputs: {$set: inputs}})

    this.setState({loading: true})

    request.post(url)
      .set('Content-Type', 'application/json')
      .send(args)
      .end((err, res) => {
        if (err) {
          this.setState({error: true, errorMessage: err.message, loading: false})
        } else {
          // now we should update the textarea with the completed version


          // console.log(res, res.text)
          var jsonObject = JSON.parse(res.text)
          // console.log(jsonObject)
          paragraphs = jsonObject.output.split("<EOP>")

          for (var paragraph of paragraphs) {
            lines = paragraph.split("<EOL>")

            for (var line of lines) {
              var l = line.trim()
              l = l.replace(/ ' /g, "'")
              l = l.replace(/ , /g, ", ")
              finalText = `${finalText}${l.charAt(0).toUpperCase()}${l.slice(1)}\n`
            }
            finalText = `${finalText}\n`
          }
          this.addEditorText(finalText)
        }
      })
  }


  addEditorText(string, loading = false) {
    const { editorState } = this.state
    const content = editorState.getCurrentContent()
    const selection = editorState.getSelection()

    const newContentState = Modifier.insertText(
      content,
      selection,
      string
    )

    var newEditorState = EditorState.push(editorState, newContentState, 'insert-fragment')

    this.setState({
      editorState: newEditorState,
      loading: loading
    })
  }


  render() {
    const { editorState } = this.state

    return (
      <div className="App">
        <h1>Editor</h1>

        <div id="editor-group">
          <Editor
            ref="editor"
            className="editor"
            editorState={editorState}
            handleBeforeInput={this.handleBeforeInput}
            onChange={this.onChange}
          />
        </div>

        <div id="options">

          <div className="form-group">
            <label htmlFor="random">
              <input type="checkbox" name="random" value={this.state.form.random} onChange={this.onFormChange}/>
              Random?
            </label>

            <label htmlFor="temperature">
              <input type="text" name="temperature" value={this.state.form.temperature} onChange={this.onFormChange}/>
              Temperature
            </label>

            <label htmlFor="topk">
              <input type="text" name="topk" value={this.state.form.topk} onChange={this.onFormChange}/>
              Top-K
            </label>

          </div>

          <div className="form-group">
            <input
              type="text"
              name="number"
              value={this.state.form.number}
              onChange={this.onFormChange}
            />

            <select name="kind" value={this.state.form.kind} onChange={this.onFormChange}>
              <option value="word">Word</option>
              <option value="sentence">Sentence</option>
              <option value="paragraph">Paragraphs</option>
            </select>
          </div>

          {this.state.error ? (
            <div className="form-error">Error: {this.state.errorMessage}</div>
          ) : ""}

          { this.state.loading ? (
            <div className="submit-form">Loading...</div>
          ) : (
            <button className="submit-form" type="submit" onClick={this.submitForm}>Valider</button>
          )}
        </div>
      </div>
    );
  }
}


export default App;
