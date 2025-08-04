# Question Answering with Transformers (BERT on SQuAD)

This project demonstrates a simple yet powerful Question Answering system using HuggingFace Transformers and the SQuAD v1.1 dataset. It evaluates a fine-tuned BERT model for extractive QA using Exact Match (EM) and F1 score.

---

## 📁 Project Structure

qa_transformers_project/
│
├── main.py # Loads model & dataset, runs predictions
├── evaluate.py # Evaluation metrics (EM, F1)
├── utils.py # Helper functions (e.g., get_predictions)
├── qa_notebook.ipynb # Interactive development and testing
├── requirements.txt # Dependencies
└── README.md # Project documentation

yaml
Copy code

---

## 🚀 How to Run

1. Clone the repo or download the files.

2. Set up a virtual environment (recommended):

```bash
python -m venv qa_env
source qa_env/bin/activate  # On Windows: qa_env\Scripts\activate
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the project:

bash
Copy code
python main.py
Or use the Jupyter notebook:

bash
Copy code
jupyter notebook qa_notebook.ipynb
📊 Model Used
bert-large-uncased-whole-word-masking-finetuned-squad
(From HuggingFace, fine-tuned for QA tasks on SQuAD)

📈 Evaluation Metrics
Exact Match (EM): Checks if the predicted answer exactly matches the expected one.

F1 Score: Harmonic mean of token-level precision and recall — useful when partial matches matter.

📦 Dataset
SQuAD v1.1
Automatically loaded using the 🤗 datasets library.

yaml
Copy code

---
