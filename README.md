# Thoughtware

This is **thoughtware**, an experimental web server where an LLM handles all application logic.

See accompanying blog post at https://samdobson.uk/posts/thoughtware/

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment:**
   Create a `.env` file:
   ```env
   LLM_PROVIDER=anthropic
   ANTHROPIC_API_KEY=sk-ant-...
   ANTHROPIC_MODEL=claude-haiku-4-5
   ```

3. **Run the server:**
   ```bash
   python server.py
   ```

4. **Visit:** http://localhost:8000

## License

MIT

## Acknowledgements

Thoughtware was inspired by the excellent [Nokode project](https://github.com/samrolken/nokode/).
