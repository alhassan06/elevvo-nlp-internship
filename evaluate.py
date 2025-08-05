from transformers import pipeline
import torch
import json
from tqdm import tqdm
from sklearn.metrics import f1_score
import re


def load_squad_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        squad = json.load(f)
    
    data = []
    for article in squad['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                if qa['answers']:  
                    answer = qa['answers'][0]['text']
                    data.append({'context': context, 'question': question, 'answer': answer})
    return data


def exact_match_score(prediction, ground_truth):
    return int(normalize_text(prediction) == normalize_text(ground_truth))


def normalize_text(s):
    return re.sub(r'\W+', ' ', s).strip().lower()


def f1(prediction, ground_truth):
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * (precision * recall) / (precision + recall)


def evaluate(qa_pipeline, dataset, max_samples=100):
    exact_matches = []
    f1_scores = []

    for sample in tqdm(dataset[:max_samples]):
        context = sample['context']
        question = sample['question']
        true_answer = sample['answer']

        result = qa_pipeline(question=question, context=context)
        predicted_answer = result['answer']

        em = exact_match_score(predicted_answer, true_answer)
        f1_score_val = f1(predicted_answer, true_answer)

        exact_matches.append(em)
        f1_scores.append(f1_score_val)

        print(f"Q: {question}")
        print(f"Predicted: {predicted_answer}")
        print(f"Expected: {true_answer}")
        print()

    print(f"Results on {len(exact_matches)} samples:")
    print(f"Exact Match (EM): {sum(exact_matches)/len(exact_matches)*100:.2f}%")
    print(f"F1 Score: {sum(f1_scores)/len(f1_scores)*100:.2f}%")


if __name__ == "__main__":
    print("Loading model...")
    device = 0 if torch.cuda.is_available() else -1
    print(f"Device set to use {'cuda' if device == 0 else 'cpu'}")

    qa_pipeline = pipeline(
        "question-answering",
        model="bert-large-uncased-whole-word-masking-finetuned-squad",
        tokenizer="bert-large-uncased-whole-word-masking-finetuned-squad",
        device=device
    )

    print("Loading SQuAD dev data...")
    dataset = load_squad_data("data/dev-v1.1.json")

    print("Evaluating...")
    evaluate(qa_pipeline, dataset, max_samples=100)  # You can increase to 1000+
