# LLM-Notebook

LLM-Notebook is a powerful and user-friendly interface for interacting with large language models through OpenRouter's API. It provides a collaborative writing space with real-time token counting, customizable sampling parameters, and an intuitive interface for managing your writing sessions.

## Features

- **Collaborative Writing Space**: A clean, distraction-free interface for writing and generating text
- **Real-time Token Counter**: Keep track of your token usage as you write
- **Multiple Model Support**: Access various language models through OpenRouter's API
- **Customizable Parameters**: Fine-tune generation with temperature, top-p, min-p, top-k, and max tokens
- **Preset Management**: Save and load your favorite parameter configurations
- **Export Functionality**: Export your writing sessions with timestamps and token counts
- **Theme Options**: Choose between light, dark, or default themes
- **Advanced Settings**: Configure port, authentication, and other system settings

## Requirements

- Python 3.7+
- OpenRouter API key
- Required Python packages:
  - gradio
  - requests
  - tiktoken

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Mithadon/LLM-Notebook.git
cd LLM-Notebook
```

2. Install required packages:
```bash
pip install gradio requests tiktoken
```

3. Get an OpenRouter API key:
   - Visit [OpenRouter](https://openrouter.ai/)
   - Sign up for an account
   - Navigate to your dashboard
   - Generate an API key

## Usage

1. Run the application:
   - On Windows: Double-click `run.bat`
   - On Linux/Mac: Execute `./run.sh`

2. Open your web browser and navigate to `http://localhost:7860` (or your configured port)

3. On first run, the application will create a `user_settings.json` file. Input your OpenRouter API key in the Settings tab.

4. Start writing in the Notebook tab and use the "Continue Writing" button to generate text

## Configuration

The application automatically creates a `user_settings.json` file on first run with default settings. You can modify:

- Port number
- Default model
- Theme
- API key
- Authentication settings
- Browser launch preferences

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
