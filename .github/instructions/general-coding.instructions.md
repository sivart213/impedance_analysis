---
applyTo: "**"
---
# Project general coding standards

**Never edit the document directly**
**When code is provided in a prompt, always prioritize the provided code for your response.**
**In ask mode, always provide explanations for your implementation and reasoning.**
**If, during a sequence of prompts, the user indicates that your response is incorrect or not addressing the issue, do not repeat the previous response. Instead, pause and request additional clarification or context from the user. If possible, indicate what information might be missing or ambiguous that could be causing the repeated failure.**

## Naming Conventions
- Use PascalCase for component names, interfaces, classes and type aliases
- Use snake_case for variables, functions, and methods
- Prefix private class members and nested functions with underscore (_)
- Use ALL_CAPS for constants

## Error Handling
- Use try/catch blocks for async operations
- Always log errors with contextual information

## Code Review Response Standards
- Always evaluate the provided code as written, without making changes unless explicitly requested.
- Clearly state if the code is correct and why.
- If you suggest improvements, put them in a separate section after your evaluation.
- Never mix code changes with your evaluation—always separate recommendations from the review.