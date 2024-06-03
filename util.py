from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, TrainingArguments
from youtube_transcript_api import YouTubeTranscriptApi
from deepmultilingualpunctuation import PunctuationModel
from googletrans import Translator
import time
import torch
import re


def load_model(cp):
    tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
    model  = AutoModelForSeq2SeqLM.from_pretrained(cp)
    return tokenizer, model 


def summarize(text, model, tokenizer, num_beams=4, device='cpu'):
    model.to(device)
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True, padding = True).to(device)
    
    with torch.no_grad():
        summary_ids = model.generate(inputs, max_length=256, num_beams=num_beams)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary


def processed(text):
    processed_text = text.replace('\n', ' ')
    processed_text = processed_text.lower()
    return processed_text


def get_subtitles(video_url):
    try:
        video_id = video_url.split("v=")[1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        subs = " ".join(entry['text'] for entry in transcript)
        print(subs)

        return transcript, subs

    except Exception as e:
        return [], f"An error occurred: {e}"

from youtube_transcript_api import YouTubeTranscriptApi


def restore_punctuation(text, model_restore):
    model = model_restore
    result = model.restore_punctuation(text)
    return result


def translate_long(text, language='vi'):
    translator = Translator()
    limit = 4700
    chunks = []
    current_chunk = ''

    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= limit:
            current_chunk += sentence.strip() + ' '
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence.strip() + ' '

    if current_chunk:
        chunks.append(current_chunk.strip())

    translated_text = ''

    for chunk in chunks:
        try:
            time.sleep(1)
            translation = translator.translate(chunk, dest=language)
            translated_text += translation.text + ' '
        except Exception as e:
            translated_text += chunk + ' '

    return translated_text.strip()

def split_into_chunks(text, max_words=800, overlap_sentences=2):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        word_count = len(sentence.split())
        if current_word_count + word_count <= max_words:
            current_chunk.append(sentence)
            current_word_count += word_count
        else:
            if len(current_chunk) >= overlap_sentences:
                overlap = current_chunk[-overlap_sentences:]
            chunks.append(' '.join(current_chunk))
            current_chunk = current_chunk[-overlap_sentences:] + [sentence]
            current_word_count = sum(len(sent.split()) for sent in current_chunk)
    if current_chunk:
        if len(current_chunk) >= overlap_sentences:
            overlap = current_chunk[-overlap_sentences:]
        chunks.append(' '.join(current_chunk))
    
    return chunks


def post_processing(text):
    sentences = re.split(r'(?<=[.!?;])\s*', text)
    for i in range(len(sentences)):
        if sentences[i]:
            sentences[i] = sentences[i][0].upper() + sentences[i][1:]
    text = " ".join(sentences)
    return text


def display(text):
    sentences = re.split(r'(?<=[.!?;])\s*', text)
    unique_sentences = list(dict.fromkeys(sentences[:-1]))
    formatted_sentences = [f"• {sentence}" for sentence in unique_sentences]
    return formatted_sentences
