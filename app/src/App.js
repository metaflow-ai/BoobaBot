import React, { Component } from 'react';
import { Editor, EditorState, Modifier } from 'draft-js';

import './App.css';
import '../node_modules/draft-js/dist/Draft.css'


class App extends Component {

  constructor(props) {
    super(props)

    this.state = {
      editorState: EditorState.createEmpty(),
    }

    this.focus = () => this.refs.editor.focus()
    this.onChange = this.onChange.bind(this)
    this.handleBeforeInput = this.handleBeforeInput.bind(this)
    this.selectProposal = this.selectProposal.bind(this)
  }


  onChange(editorState) {
    this.setState({editorState}, () => {
      this.focus()
    })
  }


  handleBeforeInput(lastChar) {
    if (lastChar === ' ') {
      this.selectProposal('ALO')
      return true
    }

    return false
  }


  selectProposal(str) {
    const { editorState } = this.state
    const content = editorState.getCurrentContent()
    const selection = editorState.getSelection()

    const newContentState = Modifier.insertText(
      content,
      selection,
      str + " "
    )

    const newEditorState = EditorState.push(editorState, newContentState, 'insert-fragment')

    this.onChange(newEditorState)
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
          <div id="buttons">
            <a href="#" onClick={this.selectProposal.bind(this, 'prop1')}>Prop 1</a>
            <a href="#" onClick={this.selectProposal.bind(this, 'prop2')}>Prop 2</a>
            <a href="#" onClick={this.selectProposal.bind(this, 'prop3')}>Prop 3</a>
          </div>
        </div>
      </div>
    );
  }
}


export default App;
