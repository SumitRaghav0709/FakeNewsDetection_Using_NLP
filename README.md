# Fake News Detection

A machine learning project for classifying news articles as **real** or **fake**. The project includes a Python app, a web interface, and a saved model.

## Project contents

- `fake_news_app.py` — Python application for running predictions.
- `index_fake_news.html` — HTML page for the web interface.
- `saved_fake_news_model/` — saved model files used by the application.
- `docs/` — project documentation.
- `test_fake_news_app.py` — tests for the application.

> Update this list if your project uses different filenames or contains additional files.

## Requirements

- Python 3
- Packages listed in `requirements.txt`

## Setup

1. Clone the repository and open its folder:

   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   macOS or Linux:

   ```bash
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Run

Start the application with:

```bash
python fake_news_app.py
```

If the app starts a local web server, open the address printed in the terminal. If it expects a separate web server or a different command, replace this section with the command used by your project.

## Tests

Run the test suite with:

```bash
python -m pytest
```

If `pytest` is not included in `requirements.txt`, install it first or use the test command documented by the project.

## Model and data

The application uses the model files in `saved_fake_news_model/`. Add information here about the training dataset, model type, and how to retrain it if those details are available. Do not commit private, licensed, or otherwise restricted datasets unless you have permission to share them.

## Limitations

Predictions are model estimates and may be incorrect. Verify claims with reliable sources before relying on them or sharing them.

## License

No license has been specified yet. Add a `LICENSE` file before redistributing the project.
