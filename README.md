# Document QA Bot
![WEBUI](./images/webui_interface.png)
## Overview
The Document QA Bot is a conversational AI application that allows users to upload Markdown files and ask questions based on the content of those files. It utilizes advanced language models to provide relevant answers and maintain a conversation history.

## Features
- Upload Markdown files for processing.
- Ask questions related to the uploaded documents.
- Choose between online and offline models for generating answers.
- View conversation history and retrieved document chunks.

## Requirements
- Python 3.x
- Gradio
- Langchain
- Ollama Wrapper
- Hugging Face Transformers

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the necessary API keys for the language models.

## Usage
1. Run the application:
   ```bash
   python document_qa_bot.py
   ```

2. Open your web browser and navigate to `http://localhost:7860`.

3. Upload a Markdown file using the provided interface.

4. Enter your question in the input box and select whether to use the online model.

5. Click the submit button to receive answers and view the conversation history.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.