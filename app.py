import os, re, pickle
import streamlit as st

st.set_page_config(page_title="Food Review Sentiment Analyzer", page_icon="🍽️", layout="centered")
st.title("🍽️ Amazon Fine Food Reviews")
st.subheader("Sentiment Analysis using CNN")
st.write("Enter a food-product review to estimate whether its sentiment is positive or negative.")

MODEL_PATH = "models/sentiment_model.keras"
TOKENIZER_PATH = "models/tokenizer.pkl"

def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^A-Za-z\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()

def demo_predict(text):
    # Transparent fallback so the UI works before the full CNN is trained.
    positive = {"good","great","excellent","love","loved","tasty","delicious","fresh","perfect","amazing","best","wonderful","enjoy","happy","recommend","yummy","nice","favorite","fantastic","smooth"}
    negative = {"bad","awful","terrible","hate","hated","stale","disgusting","poor","worst","disappointing","disappointed","waste","bland","rancid","broken","unhappy","return","refund","horrible"}
    words = set(clean_text(text).split())
    score = len(words & positive) - len(words & negative)
    if score > 0:
        return "Positive", min(0.95, 0.60 + 0.08 * score), "Demo keyword mode"
    if score < 0:
        return "Negative", min(0.95, 0.60 + 0.08 * abs(score)), "Demo keyword mode"
    return "Uncertain / mixed", 0.50, "Demo keyword mode"

def cnn_predict(text):
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    model = tf.keras.models.load_model(MODEL_PATH)
    seq = tokenizer.texts_to_sequences([clean_text(text)])
    padded = pad_sequences(seq, maxlen=200, padding="post", truncating="post")
    score = float(model.predict(padded, verbose=0)[0][0])
    return ("Positive" if score >= 0.5 else "Negative"), (score if score >= 0.5 else 1-score), "Trained CNN model"

st.info("For full CNN predictions, train the model using train_cnn.py and the Amazon Fine Food Reviews CSV. Until then, this app uses a clearly labelled demo keyword mode.")
review = st.text_area("Enter review", placeholder="Example: The coffee tastes great and I would buy it again.", height=130)
if st.button("Analyze Sentiment", type="primary", use_container_width=True):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
            try:
                label, confidence, mode = cnn_predict(review)
            except Exception as e:
                label, confidence, mode = demo_predict(review)
                st.warning("CNN model could not be loaded. Showing demo-mode result instead.")
        else:
            label, confidence, mode = demo_predict(review)
        st.metric("Predicted sentiment", label)
        st.progress(int(confidence*100))
        st.caption(f"Indicative confidence: {confidence:.0%} · Mode: {mode}")
        st.caption("This is an academic prototype; results can be wrong, especially for sarcasm, negation, or mixed opinions.")

with st.expander("How it works"):
    st.markdown("""
    1. Review text is cleaned and normalized.
    2. The trained pipeline tokenizes text and pads it to 200 tokens.
    3. A CNN learns local word-pattern features through an Embedding and Conv1D layer.
    4. The output is classified as positive or negative.

    Ratings 1–2 are labelled negative, ratings 4–5 positive, and rating 3 is excluded during training.
    """)
