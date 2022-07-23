
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from model_utils import EncoderLayer, DecoderLayer, PositionwiseFeedForward, LayerNorm, _gen_bias_mask, _gen_timing_signal, share_embedding, _get_attn_subsequent_mask, get_input_from_batch, get_output_from_batch, SoftmaxOutputLayer
import config
import pprint
pp = pprint.PrettyPrinter(indent=1)
import os


class Encoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0, layer_dropout=0,
                 attention_dropout=0.1, relu_dropout=0.1, use_mask=False, universal=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """

        super(Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if (self.universal):
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length) if use_mask else None,
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        if (self.universal):
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, mask):
        # Add input dropout
        x = self.input_dropout(inputs)

        # Project to hidden size
        x = self.embedding_proj(x)

        if (self.universal):
            if (config.act):
                x, (self.remainders, self.n_updates) = self.act_fn(x, inputs, self.enc, self.timing_signal,
                                                                   self.position_signal, self.num_layers)
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                    x = self.enc(x, mask=mask)
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

            for i in range(self.num_layers):
                x = self.enc[i](x, mask)

            y = self.layer_norm(x)
        return y


class Decoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=300, input_dropout=0, layer_dropout=0,
                 attention_dropout=0.1, relu_dropout=0.1, universal=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(Decoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if (self.universal):
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length),  # mandatory
                  None,
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)

        self.dec = nn.Sequential(*[DecoderLayer(*params) for l in range(num_layers)])

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output, mask):
        mask_src, mask_trg = mask
        dec_mask = torch.gt(mask_trg + self.mask[:, :mask_trg.size(-1), :mask_trg.size(-1)], 0)
        # Add input dropout
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)
        # Add timing signal
        x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

        # Run decoder
        y, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src, dec_mask)))

        # Final layer normalization
        y = self.layer_norm(y)
        return y, attn_dist


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.p_gen_linear = nn.Linear(config.hidden_dim, 1)

    def forward(self, x, attn_dist=None, enc_batch_extend_vocab=None, extra_zeros=None, temp=1, beam_search=False,
                attn_dist_db=None):

        logit = self.proj(x)
        return F.log_softmax(logit, dim=-1)

class RLTrans(nn.Module):

    def __init__(self, vocab, emo_number, model_file_path=None, is_eval=False, load_optim=False):
        super(RLTrans, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.embedding = share_embedding(self.vocab, config.preptrained)
        self.encoder = Encoder(config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads,
                               total_key_depth=config.depth, total_value_depth=config.depth,
                               filter_size=config.filter)

        self.r_encoder = Encoder(config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads,
                                 total_key_depth=config.depth, total_value_depth=config.depth,
                                 filter_size=config.filter)
        self.decoder = Decoder(config.emb_dim, hidden_size=config.hidden_dim, num_layers=config.hop,
                               num_heads=config.heads,
                               total_key_depth=config.depth, total_value_depth=config.depth,
                               filter_size=config.filter)
        self.bow = SoftmaxOutputLayer(config.hidden_dim, self.vocab_size)
        self.generator = Generator(config.hidden_dim, self.vocab_size)

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)

        if model_file_path is not None:
            print("loading weights")
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.r_encoder.load_state_dict(state['r_encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'])
            self.generator.load_state_dict(state['generator_dict'])
            self.embedding.load_state_dict(state['embedding_dict'])
            self.bow.load_state_dict(state['bow'])
        if (config.USE_CUDA):
            self.cuda()
        if is_eval:
            self.eval()
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
            if (load_optim):
                self.optimizer.load_state_dict(state['optimizer'])
                if config.USE_CUDA:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda()
        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

    def save_model(self, running_avg_ppl, iter):
        state = {
            'iter': iter,
            'encoder_state_dict': self.encoder.state_dict(),
            'r_encoder_state_dict': self.r_encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'generator_dict': self.generator.state_dict(),
            'embedding_dict': self.embedding.state_dict(),
            'bow': self.bow.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_ppl
        }
        model_save_path = os.path.join(self.model_dir,'model_{:.1f}_{:.4f}'.format(iter, running_avg_ppl))
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def train(self, batch, iter, train=True):
        enc_batch, _, _, enc_batch_extend_vocab, extra_zeros, _= get_input_from_batch(batch)
        dec_batch, _, _, _, _ = get_output_from_batch(batch)
        self.optimizer.zero_grad()
        ## Response encode
        mask_res = batch["posterior_batch"].data.eq(config.PAD_idx).unsqueeze(1)
        posterior_mask = self.embedding(batch["posterior_mask"])
        r_encoder_outputs = self.r_encoder(self.embedding(batch["posterior_batch"]) + posterior_mask, mask_res)
        ## Encode
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mask = self.embedding(batch["input_mask"])
        encoder_outputs = self.encoder(self.embedding(enc_batch) + emb_mask, mask_src)

        if batch["program_label"] is None:
            meta = torch.empty_like(encoder_outputs[:, 0])
        else:
            meta = self.embedding(batch["program_label"])
        # Decode
        sos_token = torch.LongTensor([config.SOS_idx] * enc_batch.size(0)).unsqueeze(1)
        if config.USE_CUDA: sos_token = sos_token.cuda()

        dec_batch_shift = torch.cat((sos_token, dec_batch[:, :-1]), 1)  # (batch, len, embedding)
        mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)
        input_vector = self.embedding(dec_batch_shift)
        input_vector[:, 0] = input_vector[:, 0] + meta
        pre_logit, attn_dist = self.decoder(input_vector, encoder_outputs, (mask_src, mask_trg))
        logit = self.generator(pre_logit, attn_dist, None,
                               extra_zeros, attn_dist_db=None)
        ## loss: NNL if ptr else Cross entropy
        loss_rec = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))
        loss = loss_rec
        aux = 0
        if (train):
            loss.backward()
            # clip gradient
            nn.utils.clip_grad_norm_(self.parameters(), config.max_grad_norm)
            self.optimizer.step()

        return loss_rec.item(), math.exp(min(loss_rec.item(), 100)), aux,

    def decoder_greedy(self, batch, max_dec_step=50):
        enc_batch, _, _, enc_batch_extend_vocab, extra_zeros, _= get_input_from_batch(batch)

        ## Encode
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mask = self.embedding(batch["input_mask"])
        encoder_outputs = self.encoder(self.embedding(enc_batch) + emb_mask, mask_src)
        ys = torch.ones(enc_batch.shape[0], 1).fill_(config.SOS_idx).long()
        if config.USE_CUDA:
            ys = ys.cuda()
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        if batch["program_label"] is None:
            meta = torch.empty_like(encoder_outputs[:, 0])
        else:
            meta = self.embedding(batch["program_label"])
        decoded_words = []
        for i in range(max_dec_step + 1):
            input_vector = self.embedding(ys)
            input_vector[:, 0] = input_vector[:, 0] + meta
            out, attn_dist = self.decoder(input_vector, encoder_outputs, (mask_src, mask_trg))

            prob = self.generator(out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None)
            _, next_word = torch.max(prob[:, -1], dim=1)
            decoded_words.append(['<EOS>' if ni.item() == config.EOS_idx else self.vocab.index2word[ni.item()] for ni in
                                  next_word.view(-1)])

            if config.USE_CUDA:
                ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
                ys = ys.cuda()
            else:
                ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == '<EOS>':
                    break
                else:
                    st += e + ' '
            sent.append(st)
        return sent