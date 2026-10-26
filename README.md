# Amazon Fine Food Reviews: Sentiment Analysis using CNN
**Student:** Huzaif Ausaf Khan | **Roll No.:** A-48

## Run the working web app
1. Install Python 3.10 or 3.11 recommended for TensorFlow compatibility.
2. Open Terminal in this project folder.
3. Create a virtual environment (optional but recommended):
   `python -m venv .venv`
4. Activate it:
   - macOS/Linux: `source .venv/bin/activate`
   - Windows: `.venv\Scripts\activate`
5. Install dependencies: `pip install -r requirements.txt`
6. Start the app: `streamlit run app.py`
7. Open the local URL shown in the terminal (usually http://localhost:8501).

The UI works immediately in **demo keyword mode**. This is a transparent fallback and is not a trained CNN.

## Train the CNN
1. Obtain the Amazon Fine Food Reviews dataset from the SNAP dataset page or another permitted source.
2. Put the CSV file named `Reviews.csv` in `data/`.
3. Run: `python train_cnn.py --csv data/Reviews.csv --sample-size 100000 --epochs 5`
4. The script reports actual test metrics and saves `models/sentiment_model.keras` and `models/tokenizer.pkl`.
5. Restart/reload Streamlit. The app will use the trained CNN when both files exist.

## Labeling
- Scores 1–2 = Negative (0)
- Scores 4–5 = Positive (1)
- Score 3 is excluded from binary training

## Important
Do not report training accuracy until `train_cnn.py` has actually completed. The app is an educational prototype, not a guaranteed sentiment detector. For deployment to a public URL, deploy this folder to Streamlit Community Cloud or another hosting service.
