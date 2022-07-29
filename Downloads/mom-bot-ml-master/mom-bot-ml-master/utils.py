import youtokentome
import string
import csv 
from rouge import Rouge
from sklearn.utils.class_weight import compute_class_weight
import random
import numpy as np
import pandas as pd
import torch
import os
import copy

from tqdm.autonotebook import tqdm
from nltk import tokenize
from nltk.translate.bleu_score import corpus_bleu

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.random.manual_seed(seed)
    torch.cuda.random.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def save_checkpoints(checkpoint, epoch_number, checkpoint_path):
    os.makedirs(checkpoint_path, exist_ok=True)
    path = os.path.join(checkpoint_path, f'checkpoint_epoch_{epoch_number + 1}.pt')
    torch.save(checkpoint, path)

def log(text):
    os.makedirs('output', exist_ok=True)
    with open(os.path.join('output','log.txt'), 'a+') as file: 
        file.write(f'{text}\n')

def clean_dataset(file_path):
    """
    Clean the data
    """
    data = pd.read_csv(file_path)
    name = os.path.basename(file_path)
    file_name = name.replace('.csv', '')
    log(f"Total {data.shape[0]} articles in {file_name} dataset")
    data = data.drop_duplicates(subset=['article'])
    log(f"Without duplicates {data.shape[0]}  in {file_name} dataset") 
    data['article'] = data['article'].apply(lambda x: x.replace(u'\xa0', u' '))
    data['article']= data['article'].apply(lambda x: x.replace(' . ', ' '))
    data['article']= data['article'].apply(lambda x: x.replace("\\", ""))
    return data

def read_news_records(file_name, shuffle=True):

    with open(file_name, "r") as file:
        records =  [{k: v for k, v in row.items()} for row in tqdm(csv.DictReader(file, skipinitialspace=True))]
    if shuffle:
        random.shuffle

    return records

def read_records(data, shuffle=True):
    records =  [{k: v for k, v in row.items()} for i, row in tqdm(data.iterrows())]
    if shuffle:
        random.shuffle

    return records

def train_bpe(records, model_path, model_type="bpe", vocab_size=20000, lower=True):
    """
    Compile a dictionary for indexing tokens
    """
    temp_file_name = "dataset/temp.txt"
    with open(temp_file_name, "w") as temp:
        for record in records:
            text, summary = record['article'], record['highlights']
            if lower:
                summary = summary.lower()
                text = text.lower()
            if not text or not summary:
                continue
            temp.write(text + "\n")
            temp.write(summary + "\n")
    youtokentome.BPE.train(data=temp_file_name, vocab_size=vocab_size, model=model_path)

def calc_scores(references, predictions, metric="all"):
    log(f"Number of examples: {len(predictions)}")
    log(f"Ref: {references[-1]}")
    log(f"Hyp: {predictions[-1]}")

    if metric in ("bleu", "all"):
        log(f"BLEU: {corpus_bleu([[r] for r in references], predictions):.2}")
    if metric in ("rouge", "all"):
        rouge = Rouge()
        scores = rouge.get_scores(predictions, references, ignore_empty=True, avg=True)
        log(f"ROUGE-1: recall {scores['rouge-1']['r']:.2}, precision {scores['rouge-1']['p']:.2}, f1_score {scores['rouge-1']['f']:.2}")
        log(f"ROUGE-2: recall {scores['rouge-2']['r']:.2}, precision {scores['rouge-2']['p']:.2}, f1_score {scores['rouge-2']['f']:.2}")
        log(f"ROUGE-l: recall {scores['rouge-l']['r']:.2}, precision {scores['rouge-l']['p']:.2}, f1_score {scores['rouge-l']['f']:.2}")

def build_oracle_summary_greedy(text, gold_summary, calc_score, lower=True, max_sentences=30):
    """
    Greedy building of oracle summary
    """
  
    gold_summary = gold_summary.lower() if lower else gold_summary
    # Split the text into sentences 
    sentences = [sentence.lower() if lower else sentence for sentence in tokenize.sent_tokenize(text)][:max_sentences]
    n_sentences = len(sentences)
    oracle_summary_sentences = set()
    
    score = -1.0
    summaries = []
    for _ in range(n_sentences):
        for i in range(n_sentences):
            if i in oracle_summary_sentences:
                continue
            current_summary_sentences = copy.copy(oracle_summary_sentences)
            # Adding some sentences to an already existing summary
            current_summary_sentences.add(i)
            
            current_summary = " ".join([sentences[index] for index in sorted(list(current_summary_sentences))]) 
            
            # Count metrics
            try:
                current_score = calc_score(current_summary, gold_summary)
            except ValueError:
                current_score = 0.0
            summaries.append((current_score, current_summary_sentences))

        # If the metrics are improved  with the addition of any sentence, then try to add more
        # Otherwise stop
        best_summary_score, best_summary_sentences = max(summaries)
        if best_summary_score <= score:
            break
        oracle_summary_sentences = best_summary_sentences
        score = best_summary_score
    oracle_summary = " ".join([sentences[index] for index in sorted(list(oracle_summary_sentences))])
    return oracle_summary, oracle_summary_sentences

def calc_single_score(pred_summary, gold_summary, rouge):
    """
    Calculaing ROUGE scores
    """
    return rouge.get_scores([pred_summary], [gold_summary], avg=True, ignore_empty=True)['rouge-2']['f']

def add_oracle_summary_to_records(records, max_sentences=30, lower=True, nrows=None):
    rouge = Rouge()
    for i, record in tqdm(enumerate(records)):
        if nrows:
            if i >= nrows:
                break
        text = record['article'] 
        summary = record['highlights']

        summary = summary.lower() if lower else summary
        sentences = [sentence.lower() if lower else sentence for sentence in tokenize.sent_tokenize(text)][:max_sentences]

        oracle_summary, sentences_indicies = build_oracle_summary_greedy(text, summary, calc_score=lambda x, y: calc_single_score(x, y, rouge),
                                                                         lower=lower, max_sentences=max_sentences)
        record["sentences"] = sentences
        record["oracle_sentences"] = list(sentences_indicies)
        record["oracle_summary"] = oracle_summary
    if nrows:
        return records[:nrows]
    else:
        return records

def custom_weights(records, device):
    labels = []
    for idx, record in enumerate(records):
        sentences = record["sentences"]
        sentences_ = record['oracle_sentences']
        labels.extend([int(i in sentences_) for i in range(len(sentences))])  
    weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return torch.tensor(weights, dtype=torch.float32).to(device)