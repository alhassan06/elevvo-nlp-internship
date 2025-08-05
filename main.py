from utils import load_squad_data
from transformers import pipeline
from tqdm import tqdm
dev_data = load_squad_data("./data/dev-v1.1.json")
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
for i, item in enumerate(tqdm(dev_data[:5])):  # Testing first 5 examples
    context = item['context']
    question = item['question']
    answer = item['answers'][0]  
    result = qa_pipeline(question=question, context=context)
    print(f"\nQ: {question}\nPredicted: {result['answer']}\nExpected: {answer}")
