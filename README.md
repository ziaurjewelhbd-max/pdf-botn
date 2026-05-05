# PDF Bot

A Streamlit-based PDF QA app using Google Gemini embeddings and LangChain.

## Local use

1. Create and activate your virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Add your Google API key to `.env`:
   ```text
   GOOGLE_API_KEY=your_api_key_here
   ```
4. Run locally:
   ```powershell
   streamlit run app.py
   ```

## Deploy online

### Option 1: Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to https://streamlit.io/cloud.
3. Connect your GitHub repo and deploy.
4. Set the secret `GOOGLE_API_KEY` in the Streamlit Cloud app settings.

### Option 2: Render / Railway / Fly.io

1. Push this repository to GitHub.
2. Create a new Python web service on the platform.
3. Use `python -m streamlit run app.py` as the start command.
4. Add environment variable `GOOGLE_API_KEY` in the service settings.

## Notes

- Your computer can be turned off once the app is deployed to a cloud host.
- Use Streamlit Cloud if you want the simplest browser deployment.
- Make sure the `.env` file is never pushed to GitHub with your secret key.
