# Copilot Instructions

## Standing Orders

### Standing Order 1
- **Trigger**: When generating or adding a comment.
- **Action**: **Never place a comment on the same line as code, instead placing it on its own line nearby. This is a critical rule and must be followed without exception.**

### Standing Order 2
- **Trigger**: When the user attaches a file or selects lines of code.
- **Action**: Reference only the selected lines of code or attached file for the response. Do not use any version of the code from history.
- **Clarification**: If no selection is visible and Standing Order 1 is triggered, ask for a selection or clarification.
  
### Standing Order 3
- **Trigger**: When removing lines of code.
- **Action**: Do not remove lines of code unless they are being refactored. Instead, comment out the affected line by adding `# ` in front of it.
  - Removed lines do not need the change indicator.

### Standing Order 4
- **Trigger**: When the user request requires changing existing code.
- **Action**: Highlight every change by identifying the changed line by inserting a `# CHANGE: <comment>` on the previous line. 
  - `<comment>` is 2 to 3 word description of the change.
  - Never place `# CHANGE: <comment>` on the same line with code, it must always be on it's own line.

### Standing Order 5
- **Trigger**: When the user requests code changes or explanations that do not affect certain functions or code blocks.
- **Action**: Summarize the unrelated and unchanged functions or code blocks using a suitable indicator.
  - Acceptable indicators include comments (e.g., `# Function unchanged`) or the `pass` statement.
  - Apply this standing order to functions and code blocks that are not relevant to the current question/request.

### Standing Order 6
- **Trigger**: When the user asks a for a general explanation of code without mentioning or referring to a particular document.
- **Action**: Answer the question without code and query if an example is desired, tayloring the example (if any) to the response.

### Standing Order 7
- **Trigger**: When the user requests a code review.
- **Action**: Review the code for potential issues, improvements, and best practices without rewriting it unless explicitly requested.
- **Clarification**: If the user requests changes, follow the existing standing orders for code modification.

### Standing Order 8
- **Trigger**: When the last sentence of a user request is `Yes/no`, `Yes or no`, or `Explain` (case insensitive).
- **Action**: Respond the the request as appropriate without implimentation.
  - If a code example is appropriate, be brief.

### Standing Order 9
- **Trigger**: When the user specifies lines of code.
- **Action**: Reference the specified lines of code for the response.

<!-- ### Standing Order 10
- **Trigger**: When the user requests a fix for a test failure.
- **Action**: Propose a fix for the test failure based on the provided information.

### Standing Order 11
- **Trigger**: When the user requests a new file or project scaffold.
- **Action**: Scaffold the requested code for a new file or project in the workspace.

### Standing Order 12
- **Trigger**: When the user requests a new Jupyter Notebook.
- **Action**: Create a new Jupyter Notebook in the workspace.

### Standing Order 13
- **Trigger**: When the user asks for an explanation of the code.
- **Action**: Provide a detailed explanation of the code, including its purpose and functionality.

### Standing Order 14
- **Trigger**: When the user asks for relevant code to a query.
- **Action**: Find and provide relevant code snippets or examples, minimizing the code enough to satisfy the query with context but no more.

### Standing Order 15
- **Trigger**: When the user asks for help with VS Code commands or terminal commands.
- **Action**: Provide the necessary commands or instructions for VS Code or the terminal.

### Standing Order 16
- **Trigger**: When the user asks for an explanation of terminal output.
- **Action**: Explain what just happened in the terminal based on the provided output.

### Standing Order 17
- **Trigger**: When the user request requires code printout.
- **Action**: Limit the printout to relevant sections with enough to ensure context but otherwise minimizing the returned code.

### Standing Order 18
- **Trigger**: When the user requests a comparison or evaluation of code.
- **Action**: Ignore Comments and Whitespace, focusing only on the functional code changes.
  - Ensure that only the actual code logic is compared.

### Standing Order 19
- **Trigger**: When the user request does not explicitly require code or code modification to answer.
- **Action**: Respond the the request as appropriate but do not immediately print code.
- **Clarification**: If your native response would have included code, ask if the code is desired.

### Standing Order 20
- **Trigger**: When the user asks for unit tests.
- **Action**: Generate unit tests for the specified code. -->
