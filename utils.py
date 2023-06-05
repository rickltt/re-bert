from torch.utils.data import Dataset
import numpy as np
import math
import os
from transformers import BertTokenizer, AutoTokenizer

class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class RetriMetric:
    def __init__(self, fname, nltk_tokenizer, stops):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()  
        self.sents = []
        self.raw_data = []
        self.nltk_tokenizer = nltk_tokenizer
        self.stops = stops

        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            text = text_left + " " + aspect + " " + text_right
            if text in self.raw_data:
                continue
            self.raw_data.append(text)

            sample = self._get_tokens(text)
            self.sents.append(sample) 
                  
        self.D = len(self.sents)
        self.avgdl = sum([len(sent)+0.0 for sent in self.sents]) / self.D
        self.f = []  # 列表的每一个元素是一个dict，dict存储着一个文档中每个词的出现次数
        self.df = {} # 存储每个词及出现了该词的文档数量
        self.idf = {} # 存储每个词的idf值
        self.k1 = 1.5
        self.b = 0.75
        self.bm25_init()
    
    def _get_tokens(self, sent):
        tokens = self.nltk_tokenizer(sent)
        sample = [ token for token in tokens if token not in self.stops]
        return sample


    def bm25_init(self):
        for sent in self.sents:
            tmp = {}
            for word in sent:
                tmp[word] = tmp.get(word, 0) + 1  # 存储每个文档中每个词的出现次数
            self.f.append(tmp)
            for k in tmp.keys():
                self.df[k] = self.df.get(k, 0) + 1
        for k, v in self.df.items():
            self.idf[k] = math.log(self.D-v+0.5)-math.log(v+0.5)

    def sim_bm25(self, sent, index):
        score = 0
        for word in sent:
            if word not in self.f[index]:
                continue
            d = len(self.sents[index])
            K = self.k1*(1-self.b+self.b*d/self.avgdl)
            fi = self.f[index][word]
            R = fi*(self.k1+1)/ (fi+ K )
            score += (self.idf[word]*R)
        return score

    def sim_tfidf(self, sent, index):
        score = 0
        for word in sent:
            if word not in self.f[index]:
                continue
            score += (self.idf[word]*self.f[index][word])
        return score
    
    def simall(self, sent, method):
        scores = []
        sent =  self._get_tokens(sent)
        for index in range(self.D):
            if method == 'bm25':
                score = self.sim_bm25(sent, index)
            else:
                score = self.sim_tfidf(sent, index)
            scores.append(score)
        return scores
    
class ReDataset(Dataset):
    def __init__(self, args, mode, tokenizer, metric, method):
        self.args = args
        fname = args.dataset_file[mode]
        self.mode = mode
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        self.topk = args.topk
        self.tokenizer = tokenizer
        self.metric = metric
        self.method = method
        self.new_data = []
        instances = []
        targets = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            polarity = int(polarity) + 1
            targets.append(polarity)
            text = text_left + " " + aspect + " " + text_right
            instances.append((text,aspect))
        self.instances = instances
        self.targets = targets

        if self.method == 'dense':
            self.retrieval_corpus = np.load("./dense/{}.npy".format(args.dataset))
            self.dense_data = np.load("./dense/{}_{}.npy".format(args.dataset, mode))
        self.init()

    def init(self):
        for i, instance in enumerate(self.instances):
            if self.method == 'dense':
                sims = []
                for re_vec in self.retrieval_corpus:
                    vec = self.dense_data[i]
                    product = np.dot(re_vec,vec)
                    sims.append(product)          
            else:
                sims = self.metric.simall(instance[0], self.method)

            sorted_idx = sorted(range(len(sims)), key=lambda k: sims[k], reverse=True)
            if self.mode == 'train':
                sorted_idx = sorted_idx[1:self.topk+1]
            else:
                sorted_idx = sorted_idx[:self.topk]
            
            new_sent = instance[0]
            for idx in sorted_idx:
                new_sent += self.metric.raw_data[idx]

            aspect = instance[1]
            polarity = self.targets[i]

            concat_bert_indices = self.tokenizer.text_to_sequence("[CLS] " + new_sent + " [SEP]" + aspect + " [SEP]")
            text_indices = self.tokenizer.text_to_sequence(new_sent)
            text_len = np.sum(text_indices != 0)
            aspect_indices = self.tokenizer.text_to_sequence(aspect)
            aspect_len = np.sum(aspect_indices != 0)

            concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
            concat_segments_indices = pad_and_truncate(concat_segments_indices, self.tokenizer.max_seq_len)

            attention_length = text_len + aspect_len + 3
            if attention_length > self.tokenizer.max_seq_len:
                attention_mask = [1] * self.tokenizer.max_seq_len
            else:
                attention_mask = [1] * (attention_length) + [0] * (self.tokenizer.max_seq_len - attention_length)

            attention_mask = np.array(attention_mask)

            item = {
                'concat_bert_indices': concat_bert_indices,
                'concat_segments_indices': concat_segments_indices,
                'attention_mask': attention_mask,
                'polarity': polarity,
            }  

            self.new_data.append(item)

    def __getitem__(self, index):
        return self.new_data[index] 
    def __len__(self):
        return len(self.new_data)  


