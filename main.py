from utils import load_squad_data
from transformers import pipeline
from tqdm import tqdm

# Load the dev dataset
dev_data = load_squad_data("./data/dev-v1.1.json")
 

# Load the pretrained QA pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Evaluate a few examples
for i, item in enumerate(tqdm(dev_data[:5])):  # Testing first 5 examples
    context = item['context']
    question = item['question']
    answer = item['answers'][0]  # Ground truth
    result = qa_pipeline(question=question, context=context)
    print(f"\nQ: {question}\nPredicted: {result['answer']}\nExpected: {answer}")
