from flask import Flask, request, render_template
import os
from rag_engine import process_pdf_and_query

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = ""
    if request.method == 'POST':
        uploaded_file = request.files['pdf']
        question = request.form['question']
        if uploaded_file.filename != '':
            pdf_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
            uploaded_file.save(pdf_path)
            answer = process_pdf_and_query(pdf_path, question)
    return render_template('index.html', answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
