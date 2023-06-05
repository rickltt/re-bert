import torch
import torch.nn as nn
from transformers import RobertaModel

class RE_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(RE_BERT, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(bert.config.hidden_size, opt.polarities_dim)

    def forward(self, concat_bert_indices, concat_segments_indices, attention_mask):
        #text_bert_indices, bert_segments_ids, attention_mask = inputs[0], inputs[1], inputs[2]
        if type(self.bert) == RobertaModel:
            _, pooled_output = self.bert(concat_bert_indices)
        else:
            _, pooled_output = self.bert(input_ids = concat_bert_indices, attention_mask = attention_mask, token_type_ids=concat_segments_indices)

        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits