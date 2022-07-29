import argparse
import sys
sys.setrecursionlimit(1500)

import nltk
nltk.download('punkt')


from dataloader import Dataloader
from models import SentenceTaggerRNN
from train_model import *
from utils import *

def main(num_samples,
         batch_size,
         num_epochs,
         clip,
         seed):

    seed_everything(seed)

    train = clean_dataset('')
    val = clean_dataset('')
    test = clean_dataset('')
    train_records = read_records(train, shuffle=True)
    val_records = read_records(val, shuffle=False)
    test_records = read_records(test, shuffle=False)

    log('Training the BPE tokenizer...')
    train_bpe(train_records, "BPE_model.bin")
    log('Done')
    bpe_tokenizer = youtokentome.BPE('BPE_model.bin')
    vocabulary = bpe_tokenizer.vocab()

    #Cache oracle summary to RAM
    ext_train_records = add_oracle_summary_to_records(train_records, nrows=num_samples) 
    ext_val_records = add_oracle_summary_to_records(val_records, nrows=num_samples)
    ext_test_records = add_oracle_summary_to_records(test_records, nrows=num_samples)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator = Dataloader(ext_train_records, vocabulary, batch_size, bpe_tokenizer, device=device)
    val_iterator = Dataloader(ext_val_records, vocabulary, batch_size, bpe_tokenizer, device=device)
    test_iterator = Dataloader(ext_test_records, vocabulary, batch_size, bpe_tokenizer, device=device)

    vocab_size = len(vocabulary)
    model = SentenceTaggerRNN(vocab_size).to(device)

    params_count = np.sum([p.numel() for p in model.parameters() if p.requires_grad])
    log("Trainable params: {}".format(params_count))

    log('Training the model...')
    training(model, ext_train_records, train_iterator, val_iterator, device, num_epochs, clip, use_class_weights=True)

    assert len(ext_train_records) >= len(test_iterator) * batch_size, "Not enough examples"
    
    log('Evaluating the model quality on the test data...')
    inference_summarunner_without_novelty_with_class_weights = inference_summarunner(model, test_iterator, top_k=3, threshold=None)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=1000, required=True, help='Number of examples')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--clip', type=float, default=1.0, help='Value for gradient clipping')
    parser.add_argument('--seed', type=int, default=42, help='A seed value')

    args = parser.parse_args()
    main(**vars(args))