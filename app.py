from util import *
import streamlit as st
from deepmultilingualpunctuation import PunctuationModel

cp_aug = 'minnehwg/finetune-newwiki-summarization-ver-augmented'


def get_model(cp):
    checkpoint = cp
    tokenizer, model = load_model(checkpoint)
    return tokenizer, model

tokenizer, model = get_model(cp_aug)
restore_model = PunctuationModel()

def execute_func(url, model, tokenizer, punc_model):
    trans, sub = get_subtitles(url)
    sub = restore_punctuation(sub, punc_model)
    vie_sub = translate_long(sub)
    vie_sub = processed(vie_sub)
    chunks = split_into_chunks(vie_sub, 700, 2)
    sum_para = []
    for i in chunks:
        tmp = summarize(i, model, tokenizer, num_beams=3)
        sum_para.append(tmp)
    suma = ''.join(sum_para)
    del sub, vie_sub, sum_para, chunks
    suma = post_processing(suma)
    re = display(suma)
    return re

def generate_summary(url):
    results = execute_func(url, model, tokenizer, restore_model)
    summary = "\n".join(results)
    return summary

def generate_summary_and_video(url):
    summary = generate_summary(url)
    summary_html = summary.replace("\n", "<br>")
    try:
        video_id = url.split("v=")[1].split("&")[0]
        iframe = f'<iframe width="300" height="200" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'
        return f"{iframe}<br><br>Những ý chính trong video:<br><br>{summary_html}"
    except IndexError:
        return f"**Summary:**\n{summary}\n\nInvalid YouTube URL for video display."


st.title("Chào mừng đến với hệ thống tóm tắt của Minne >.< ")
tokenizer, model = get_model()
input_text = st.text_area("Enter your URL:")
    
if st.button("Generate"):
    generate_summary_and_video(url)
    st.text_area("Kết quả tóm tắt:", value=output_text, height=400)
