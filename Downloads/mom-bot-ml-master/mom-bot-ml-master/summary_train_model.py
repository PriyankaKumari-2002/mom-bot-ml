import time
import math
import torch
import os
import re
import tqdm
from nltk.tokenize import WordPunctTokenizer
import matplotlib
matplotlib.rcParams.update({'figure.figsize': (16, 12), 'font.size': 14})
import matplotlib.pyplot as plt
from IPython.display import clear_output

from utils import *

def train_epoch(model, iterator, optimizer, criterion, clip, train_history=None, valid_history=None):
    os.makedirs('output', exist_ok=True)

    model.train()
    epoch_loss = 0
    history = []
    for i, batch in enumerate(iterator):
        
        logits = model(batch["inputs"])
        logits = torch.cat([-logits.unsqueeze(2), logits.unsqueeze(2)], dim=2)
        loss = criterion(logits.permute(1,2,0), batch["outputs"].permute(1,0))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        history.append(loss.item())
        if (i+1)%10==0:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

            clear_output(True)
            ax[0].plot(history, label='train loss')
            ax[0].set_xlabel('Batch')
            ax[0].set_title('Train loss')
            if train_history is not None:
                ax[1].plot(train_history, label='general train history')
                ax[1].set_xlabel('Epoch')
            if valid_history is not None:
                ax[1].plot(valid_history, label='general valid history')
            plt.legend()
            plt.savefig(os.path.join('output', 'training_history.png'))
            plt.close()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    
    epoch_loss = 0
    history = []
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            logits = model(batch["inputs"])
            logits = torch.cat([-logits.unsqueeze(2), logits.unsqueeze(2)], dim=2)
            loss = criterion(logits.permute(1,2,0), batch["outputs"].permute(1,0))
        
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_with_logs(model, train_iterator, valid_iterator, optimizer, criterion, num_epochs, clip):
    train_history = []
    valid_history = []
    
    
    best_valid_loss = float('inf')
    
    for epoch in range(num_epochs):
        
        log(f'Training epoch: {epoch+1:02}') 
        start_time = time.time()
        
        train_loss = train_epoch(model, train_iterator, optimizer, criterion, clip, train_history, valid_history)
        valid_loss = evaluate(model, valid_iterator, criterion)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        checkpoint = {'epoch': epoch + 1,
                      'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict()
                      }
        save_checkpoints(checkpoint, epoch, checkpoint_path='checkpoints')
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(checkpoint, f'checkpoints/best_validation_model.pt')
        
        train_history.append(train_loss)
        valid_history.append(valid_loss)
        log(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        log(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        log(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        
def punct_detokenize(text):
    text = text.strip()
    punctuation = ",.!?:;%"
    closing_punctuation = ")]}"
    opening_punctuation = "([}"
    for ch in punctuation + closing_punctuation:
        text = text.replace(" " + ch, ch)
    for ch in opening_punctuation:
        text = text.replace(ch + " ", ch)
    res = [r'"\s[^"]+\s"', r"'\s[^']+\s'"]
    for r in res:
        for f in re.findall(r, text, re.U):
            text = text.replace(f, f[0] + f[2:-2] + f[-1])
    text = text.replace("' s", "'s").replace(" 's", "'s")
    text = text.strip()
    return text


def postprocess(ref, hyp, is_multiple_ref=False, detokenize_after=False, tokenize_after=True):
    tokenizer = WordPunctTokenizer()
    if is_multiple_ref:
        reference_sents = ref.split(" s_s ")
        decoded_sents = hyp.split("s_s")
        hyp = [w.replace("<", "&lt;").replace(">", "&gt;").strip() for w in decoded_sents]
        ref = [w.replace("<", "&lt;").replace(">", "&gt;").strip() for w in reference_sents]
        hyp = " ".join(hyp)
        ref = " ".join(ref)
    ref = ref.strip()
    hyp = hyp.strip()
    if detokenize_after:
        hyp = punct_detokenize(hyp)
        ref = punct_detokenize(ref)
    if tokenize_after:
        hyp = hyp.replace("@@UNKNOWN@@", "<unk>")
        hyp = " ".join([token for token in tokenizer.tokenize(hyp)])
        ref = " ".join([token for token in tokenizer.tokenize(ref)])
    return ref, hyp

def training(model, records, train_iterator, val_iterator, device, num_epochs, clip, use_class_weights, lr=1e-5):
    optimizer = torch.optim.Adam(model.parameters(), lr)

    if use_class_weights:
        # weights depend on the number of objects of class 0 and 1
        class_weights = custom_weights(records, device) #as class distribution
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights,ignore_index=2)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=2)

    train_with_logs(model, train_iterator, val_iterator, optimizer, criterion, num_epochs, clip)

def inference_summarunner(model, iterator, top_k=None, threshold=0):
    """
    Generate the extractive summaries
    """
    references = []
    predictions = []

    model.eval()
    
    for batch in iterator:

        logits = model(batch['inputs'])
        if top_k:
            sum_in = torch.argsort(logits, dim=1)[:, -top_k:] 
        else:
            sum_in = (logits > threshold).nonzero(as_tuple=False)

        for i in range(len(batch['outputs'])):
           
            summary = batch['records'][i]['highlights'].lower()
            pred_summary = ' '.join([batch['records'][i]['sentences'][ind] for ind in sum_in.sort(dim=1)[0][i] if ind < len(batch["records"][i]["sentences"])])
            summary, pred_summary = postprocess(summary, pred_summary)
            references.append(summary)
            predictions.append(pred_summary)

    calc_scores(references, predictions)

def inference(model, text, device, top_k=None, threshold=0):
    """
    Generate the extractive summary
    """
    model.eval()
    logits = model(text['inputs'])
    if top_k:
        sum_in = torch.argsort(logits, dim=1)[:, -top_k:] 
    else:
        sum_in = (logits > threshold).nonzero(as_tuple=False)
        
    pred_summary = ' '.join([text['records'][ind] for ind in sum_in.sort(dim=1)[0][0] if ind < len(text["records"])]) 
    return pred_summary