import streamlit as st
import whisper
import torch
from googletrans import Translator
from gtts import gTTS
import os
from io import BytesIO
from pydub import AudioSegment
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

@st.cache_resource
def load_model():
    try:
        return whisper.load_model("base")  
    except Exception as e:
        st.error(f"Error loading Whisper model: {str(e)}")
        return None

model = load_model()
translator = Translator()


LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
    "Chinese (Simplified)": "zh-cn",
    "Arabic": "ar",
    "Russian": "ru",
    "Japanese": "ja",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Bhojpuri": "hi",  
    "Bihari": "hi" 
}

st.title("Multilingual Speech Recognition & Transliteration")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
target_language = st.selectbox("Select target language", list(LANGUAGES.keys()))

if uploaded_file is not None and model is not None:
    with st.spinner("Processing..."):
        try:
            
            audio_data = BytesIO(uploaded_file.read())

           
            audio = AudioSegment.from_file(audio_data)
            audio = audio.set_frame_rate(16000).set_channels(1)  

           
            if audio.dBFS == float("-inf"):  
                st.error("Error: Audio file is silent or too low in volume.")
            else:
                temp_audio_path = "temp.wav"
                audio.export(temp_audio_path, format="wav")

              
                if os.path.getsize(temp_audio_path) == 0:
                    st.error("Error: Uploaded audio file is empty or corrupt.")
                else:
                   
                    result = model.transcribe(temp_audio_path)
                    original_text = result.get("text", "").strip()

                   
                    if not original_text:
                        st.error("No speech detected. Please upload a clearer audio file.")
                    else:
                        lang_code = LANGUAGES[target_language]
                        translated_text = translator.translate(original_text, dest=lang_code).text

                       
                        hinglish_text = ""
                        if target_language == "Hindi":
                            hinglish_text = transliterate(translated_text, sanscript.DEVANAGARI, sanscript.ITRANS)
                            hinglish_text = hinglish_text.lower().capitalize() 

                      
                        try:
                            tts = gTTS(translated_text, lang=lang_code)
                            tts_path = "output.mp3"
                            tts.save(tts_path)

                          
                            st.subheader("Transcribed Text")
                            st.write(original_text)

                            st.subheader(f"Translated Text ({target_language})")
                            st.write(translated_text)

                            if target_language == "Hindi":
                                st.subheader("Hinglish Transliteration")
                                st.write(hinglish_text)

                            st.subheader("Translated Speech")
                            st.audio(tts_path, format="audio/mp3")

                        except Exception as e:
                            st.error(f"TTS Error: {str(e)}")

                
                os.remove(temp_audio_path)
                if os.path.exists("output.mp3"):
                    os.remove("output.mp3")

        except torch.cuda.OutOfMemoryError:
            st.error("CUDA Out of Memory: Try reducing the model size (e.g., `base`) or using CPU mode.")
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
