import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl




import os
import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import BertTokenizer, BertModel, BertConfig
from optim_schedule import ScheduledOptim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model import SoftMaskedBert
class LitAutoEncoder(pl.LightningModule):

    def __init__(self, bert, tokenizer, device, hidden=256, layer_n=1,
                 lr=2e-5, gama=0.8, betas=(0.9, 0.999), weight_decay=0.01, warmup_steps=10000):

        super().__init__()
        self.dd = device
        self.tokenizer = tokenizer
        self.model = SoftMaskedBert(bert, self.tokenizer, hidden, layer_n, self.dd).to(self.dd)

        # if torch.cuda.device_count() > 1:
        #     print("Using %d GPUS for train" % torch.cuda.device_count())
        #     self.model = nn.DataParallel(self.model, device_ids=[0,1,2])


        self.criterion_c = nn.NLLLoss()
        self.criterion_d = nn.BCELoss()
        self.gama = gama
        self.log_freq = 5

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        out, prob = self.model(x["input_ids"], x["input_mask"], x["segment_ids"])
        return out, prob

    def training_step(self, batch, batch_idx):
        # opt = self.optimizers(use_pl_optimizer=True)
        
        lr_step = self.lr_schedulers()
        # training_step defines the train loop. It is independent of forward
        data = {key: value.to(self.device) for key, value in batch.items()}

        out, prob = self.model(data["input_ids"], data["input_mask"],
                               data["segment_ids"])  # prob [batch_size, seq_len, 1]
        prob = prob.reshape(-1, prob.shape[1])

        loss_d = self.criterion_d(prob, data['label'].float())
        loss_c = self.criterion_c(out.transpose(1, 2), data["output_ids"])
        loss = self.gama * loss_c + (1 - self.gama) * loss_d

        self.log('train_loss', loss)

        self.opt.zero_grad()
        loss.backward(retain_graph=True)
        lr_step.step()

        # return loss

    def configure_optimizers(self):
        optim = Adam(self.parameters(), lr=2e-5, betas=(0.9, 0.999), weight_decay=0.01)
        # optim_schedule = ScheduledOptim(optim, 256, n_warmup_steps=10000)
        return optim



class BertDataset(Dataset):
    def __init__(self, tokenizer, dataset, max_len=512, pad_first=True, mode='train'):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_len = max_len
        self.data_size = len(dataset)
        self.pad_first = pad_first
        self.mode = mode

    def __len__(self):
        return self.data_size

    def __getitem__(self, item):
        item = self.dataset.iloc[item]
        input_ids = item['random_text']
        input_ids = ['[CLS]'] + list(input_ids)[:min(len(input_ids), self.max_len - 2)] + ['[SEP]']
        # convert to bert ids
        input_ids = self.tokenizer.convert_tokens_to_ids(input_ids)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        pad_len = self.max_len - len(input_ids)
        if self.pad_first:
            input_ids = [0] * pad_len + input_ids
            input_mask = [0] * pad_len + input_mask
            segment_ids = [0] * pad_len + segment_ids
        else:
            input_ids = input_ids + [0] * pad_len
            input_mask = input_mask + [0] * pad_len
            segment_ids = segment_ids + [0] * pad_len

        output = {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
        }

        if self.mode == 'train':
            output_ids = item['origin_text']
            label = item['label']
            label = [int(x) for x in label if x != ' ']
            output_ids = ['[CLS]'] + list(output_ids)[:min(len(output_ids), self.max_len - 2)] + ['[SEP]']
            label = [0] + label[:min(len(label), self.max_len - 2)] + [0]

            output_ids = self.tokenizer.convert_tokens_to_ids(output_ids)
            pad_label_len = self.max_len - len(label)
            if self.pad_first:
                output_ids = [0] * pad_len + output_ids
                label = [0] * pad_label_len + label
            else:
                output_ids = output_ids + [0] * pad_len
                label = label + [0] * pad_label_len

            output = {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'output_ids': output_ids,
                'label': label
            }
        return {key: torch.tensor(value) for key, value in output.items()}


if __name__ == '__main__':
    bert_path = r'voidful/albert_chinese_tiny'

    dataset = pd.read_csv('data/processed_data/all_same_765376/train.csv')
    # dataset = pd.read_csv(r'data/processed_data/my/real_val.csv')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = BertConfig.from_pretrained(bert_path if True else 'data/chinese_wwm_pytorch/bert_config.json')
    tokenizer = BertTokenizer.from_pretrained(bert_path if True else 'data/chinese_wwm_pytorch/vocab.txt')
    # kf = KFold(n_splits=5, shuffle=True)
    # for k, (train_index, val_index) in enumerate(kf.split(range(len(dataset)))):

    from sklearn.model_selection import train_test_split

    train_index, val_index = train_test_split(
        range(len(dataset)), random_state=2019, test_size=0.10)
    k = 'no'
    print('Start train {} ford'.format(k))
    bert = BertModel.from_pretrained(bert_path if True else 'data/chinese_wwm_pytorch/pytorch_model.bin', config=config)

    train = dataset.iloc[train_index]
    val = dataset.iloc[val_index]
    train = BertDataset(tokenizer, train, max_len=152)
    train = DataLoader(train, batch_size=8, num_workers=2)
    val = BertDataset(tokenizer, val, max_len=152)
    val = DataLoader(val, batch_size=8, num_workers=2)

    autoencoder = LitAutoEncoder(bert, tokenizer, device)
    trainer = pl.Trainer()
    trainer.fit(autoencoder, train, val)
