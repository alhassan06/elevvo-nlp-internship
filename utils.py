import json

def load_squad_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        squad_dict = json.load(f)

    data = []
    for group in squad_dict['data']:
        for paragraph in group['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                answers = [ans['text'] for ans in qa['answers']]
                data.append({'context': context, 'question': question, 'answers': answers})
    return data
