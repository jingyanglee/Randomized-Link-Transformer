
import os
import sys
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import jsonlines
import config
import torch
import pprint
pp = pprint.PrettyPrinter(indent=1)
from beam_omt import Translator
from preprocess_dd import Lang, create_data, Dataset, loadLines, collate_fn
from torch.utils.data import DataLoader
from RLTransformer import RLTrans
vocab = Lang()
batch_size = config.batch_size

raw_train_data = loadLines(os.path.join(sys.path[0]+"/ParlAI/data/dailydialog/train.json"))
raw_valid_data = loadLines(os.path.join(sys.path[0]+"/ParlAI/data/dailydialog/valid.json"))
raw_test_data = loadLines(os.path.join(sys.path[0]+"/ParlAI/data/dailydialog/test.json"))

train_data = create_data(raw_train_data, vocab)
valid_data = create_data(raw_valid_data, vocab)
test_data = create_data(raw_test_data, vocab)

test_dataset = Dataset(test_data, vocab)

test_dataloader = DataLoader(dataset=test_dataset ,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=collate_fn)

def generate(model, data):
    with torch.no_grad():
        with jsonlines.open(config.json_file, mode='w') as writer:
            t = Translator(model, model.vocab)
            for j, batch in enumerate(data):
                sent_g = model.decoder_greedy(batch, max_dec_step=config.max_dec_step)
                for i, greedy_sent in enumerate(sent_g):

                    result = {'context':batch['input_txt'][i],
                        'res':greedy_sent,
                        'ref': batch["target_txt"][i]}
                    writer.write(result)
                    print("----------------------------------------------------------------------")
                    print("----------------------------------------------------------------------")
                    print("dialogue context:")
                    print(batch['input_txt'][i])
                    print("Beam: {}".format(greedy_sent))
                    print("Ref:{}".format(batch["target_txt"][i]))
                    print("----------------------------------------------------------------------")
                    print("----------------------------------------------------------------------")


# Build model, optimizer, and set states
print("Test model",config.model)
model = RLTrans(vocab, emo_number = None, model_file_path='./RL Transformer/save/model_43.3_15.0000', is_eval=False)


#generate
val_losses = []
val_ppls = []
test_generator = iter(test_dataloader)
generate(model, test_generator)
