from flask import  Flask, render_template, request
import re
from transformers import pipeline
import json
import random
import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('test_answer.json', 'r') as file:
    data = json.load(file)

summarizer = pipeline("summarization", model="basic-go/FRED-T5-large-habr-summarizer", device=device)
clean_expr = re.compile(r"[\xa0\x1a\x16\x1b\x17\x15\u2004]")
spaces_expr = re.compile(r"\s{2,}")

def process_text(text: str) -> str:
    text = clean_expr.sub(" ", text)
    text = spaces_expr.sub(" ", text)

    if "." in text:
        index = text.rindex(".")
        text = text[:index + 1]

    return text

def get_Summarize (ARTICLE):
  ARTICLE = process_text(ARTICLE)

  response = summarizer(ARTICLE, max_new_tokens=360, num_beams=2, do_sample=True, top_k=100,
    repetition_penalty=2.5, length_penalty=1.0)

  summary = process_text(response[0]["summary_text"])
  return summary

app = Flask(__name__)

@app.route("/", methods = ["GET", "POST"])
def home():
    summar_text = None
    if request.method == "POST":
        ARTICLE = request.form.get("textTo")
        summar_text = get_Summarize (ARTICLE)    
        return render_template("results.html", origin_text = ARTICLE, result_text = summar_text)
    return render_template("home.html")

@app.route("/example_result", methods = ["GET", "POST"])
def example_result():
    global data
    size = len(data["origin"])-1
    idx = str(random.randint(0,size))
    origin = data["origin"][idx]
    summar = data["summar"][idx]
    return render_template("results.html", origin_text = origin, result_text = summar)

if __name__ == "__main__":
    app.run(debug = True)
        