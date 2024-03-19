import streamlit as st
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
from wit import Wit

# Initialize Wit.ai client
wit_client = Wit("YOUR_WIT_KEY")

# Initialize the BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name)
labels = ["non-blackmail", "blackmail"]

def transcribe_audio(audio_content):
    resp = wit_client.speech(audio_content, {'Content-Type': 'audio/wav'})
    return resp["text"]

def check_blackmail(text):
    # Tokenize input text
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="tf")

    # Make prediction
    outputs = model(inputs)
    predictions = tf.nn.softmax(outputs.logits, axis=-1)

    # Get predicted label and score
    predicted_label_index = tf.argmax(predictions, axis=1).numpy()[0]
    predicted_label = labels[predicted_label_index]
    score = predictions[0][predicted_label_index].numpy()

    return predicted_label, score

def main():
    st.title("Audio Transcription and Blackmail Detection App")

    # Upload audio file
    audio_file = st.file_uploader("Upload audio file", type=["wav"])

    if audio_file is not None:
        # Transcribe audio on button click
        if st.button("Predict"):
            # Display audio
            st.audio(audio_file, format="audio/wav")
                        
            # Transcribe audio
            transcribed_text = transcribe_audio(audio_file.read())
            st.write("Transcribed Text:", transcribed_text)

            # Check for blackmail context
            predicted_label, score = check_blackmail(transcribed_text)

            if predicted_label == "blackmail":
                st.warning("Warning: This text may contain blackmail or threatening language.")
            else:
                st.success("No blackmail detected.")

if __name__ == "__main__":
    main()
