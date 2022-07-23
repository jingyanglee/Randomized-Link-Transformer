
from RLTransformer import RLTrans
from model_utils import evaluate
import config
import torch
from tqdm import tqdm
import os
import sys
from preprocess_dd import Lang, create_data, Dataset, loadLines, collate_fn
from torch.utils.data import DataLoader
torch.autograd.set_detect_anomaly(True)

vocab = Lang()
batch_size = config.batch_size

raw_train_data = loadLines(os.path.join(sys.path[0]+"/ParlAI/data/dailydialog/train.json"))
raw_valid_data = loadLines(os.path.join(sys.path[0]+"/ParlAI/data/dailydialog/valid.json"))
raw_test_data = loadLines(os.path.join(sys.path[0]+"/ParlAI/data/dailydialog/test.json"))

train_data = create_data(raw_train_data, vocab)
valid_data = create_data(raw_valid_data, vocab)
test_data = create_data(raw_test_data, vocab)

train_dataset = Dataset(train_data, vocab)
valid_dataset = Dataset(valid_data, vocab)
test_dataset = Dataset(test_data, vocab)

train_dataloader = DataLoader(dataset=train_dataset ,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=collate_fn)
valid_dataloader = DataLoader(dataset=valid_dataset ,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=collate_fn)

# Build model, optimizer, and set states
model = RLTrans(vocab, emo_number=None)
for n, p in model.named_parameters():
    if p.dim() > 1 and (n != "embedding.lut.weight" and config.preptrained):
        torch.nn.init.xavier_uniform_(p)

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(torch.cuda.current_device()))

# meta early stop
max_steps = len(train_dataset)/batch_size
patience = 3
best_loss = 10000000
best_ppl = 10000000
stop_count = 0
best_d1 = 0
# Main loop
for epoch in range(config.epochs):
    print('epoch:', epoch)
    val_losses = []
    val_ppls = []
    batch_loss = 0
    train_generator = iter(train_dataloader)
    valid_generator = iter(valid_dataloader)
    for batch_id, batch in enumerate(tqdm(train_generator)):
        loss, ppl, bow = model.train(batch, iter=epoch)
    val_loss, val_ppl, bow_val, d1, d2, d3 = evaluate(model, valid_dataloader,ty="valid", max_dec_step=50)
    print('valid loss:', val_loss)
    print('valid ppl:', val_ppl)
    if stop_count >= patience:
        print('DONE')
        break
    if val_loss < best_loss or val_loss == 'nan':
        best_loss = val_loss
        best_ppl = val_ppl
        stop_count = 0
        model.save_model(epoch,val_ppl)
    else:
        stop_count += 1
