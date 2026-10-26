"""
Train a CNN on the Amazon Fine Food Reviews dataset.
Expected CSV columns: Score and Text (the common SNAP/Kaggle dataset format).
Run: python train_cnn.py --csv data/Reviews.csv --sample-size 100000
"""
import argparse, os, re, pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models

def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^A-Za-z\\s']", " ", text)
    return re.sub(r"\\s+", " ", text).strip().lower()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/Reviews.csv")
    parser.add_argument("--sample-size", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=5)
    args=parser.parse_args()
    if not os.path.exists(args.csv):
        raise SystemExit(f"Dataset not found: {args.csv}. Download the Amazon Fine Food Reviews CSV and place it there.")
    df=pd.read_csv(args.csv, usecols=["Score","Text"])
    df=df.dropna(subset=["Score","Text"])
    df=df[df["Score"].isin([1,2,4,5])].copy()
    df["sentiment"]=(df["Score"]>=4).astype(int)
    if args.sample_size and len(df)>args.sample_size:
        df=df.groupby("sentiment", group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), max(1, args.sample_size//2)), random_state=42)
        ).reset_index(drop=True)
    df["clean"]=df["Text"].map(clean_text)
    X_train, X_test, y_train, y_test=train_test_split(
        df["clean"].astype(str), df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"]
    )
    tokenizer=Tokenizer(num_words=20000, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    x_train=pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=200, padding="post", truncating="post")
    x_test=pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=200, padding="post", truncating="post")
    model=models.Sequential([
        layers.Input(shape=(200,)),
        layers.Embedding(input_dim=20000, output_dim=128),
        layers.Conv1D(128, 5, activation="relu"),
        layers.GlobalMaxPooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
    model.fit(x_train, np.asarray(y_train), validation_split=0.1, epochs=args.epochs, batch_size=128, verbose=1)
    loss, acc=model.evaluate(x_test, np.asarray(y_test), verbose=0)
    preds=(model.predict(x_test, verbose=0).ravel()>=0.5).astype(int)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=["Negative","Positive"]))
    print("Confusion matrix:\\n", confusion_matrix(y_test, preds))
    os.makedirs("models", exist_ok=True)
    model.save("models/sentiment_model.keras")
    with open("models/tokenizer.pkl","wb") as f: pickle.dump(tokenizer,f)
    print("Saved model and tokenizer under models/")

if __name__=="__main__":
    main()
