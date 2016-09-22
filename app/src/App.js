import React, { Component } from 'react';
import { Editor, EditorState, RichUtils, Modifier } from 'draft-js';

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
    const end = selection.getEndOffset()

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

        <button onClick={this.selectProposal.bind(this, 'prop1')}>Prop 1</button>
        <button onClick={this.selectProposal.bind(this, 'prop2')}>Prop 2</button>
        <button onClick={this.selectProposal.bind(this, 'prop3')}>Prop 3</button>
        <div className="editor">
          <Editor
            ref="editor"
            editorState={editorState}
            handleBeforeInput={this.handleBeforeInput}
            onChange={this.onChange}
          />
        </div>
      </div>
    );
  }
}


export default App;
