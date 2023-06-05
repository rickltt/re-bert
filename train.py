import argparse
import torch
import torch.nn as nn
import numpy as np
import random
import os
from sklearn import metrics
from torch.utils.data import DataLoader, random_split
from transformers import AutoModel, BertModel, RobertaModel
from model import RE_BERT
from utils import Tokenizer4Bert, ReDataset, RetriMetric
from nltk import word_tokenize
from torch.nn import DataParallel
from nltk.corpus import stopwords
import logging
import sys
import math

stops = stopwords.words('english')
stops.extend([',','.','``'])

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def reset_params(model):
    model_list = [BertModel, RobertaModel]
    for child in model.children():
        if type(child) not in model_list:  # skip bert params
            for p in child.parameters():
                if p.requires_grad:
                    if len(p.shape) > 1:
                        torch.nn.init.xavier_uniform_(p)
                    else:
                        stdv = 1. / math.sqrt(p.shape[0])
                        torch.nn.init.uniform_(p, a=-stdv, b=stdv)

def train(args, model, criterion, optimizer, train_dataloader, val_dataloader):
    max_val_acc = 0
    max_val_f1 = 0
    max_val_epoch = 0
    global_step = 0
    path = None
    for i_epoch in range(args.num_epoch):
        logger.info('>' * 100)
        logger.info('epoch: {}'.format(i_epoch))
        n_correct, n_total, loss_total = 0, 0, 0
        # switch model to training mode
        model.train()
        for _, batch in enumerate(train_dataloader):
            global_step += 1
            # clear gradient accumulators
            optimizer.zero_grad()
            # inputs = [batch[col].to(args.device) for col in args.inputs_cols]
            batch = tuple(t.to(args.device) for t in batch.values())
            inputs = {
                "concat_bert_indices":batch[0],
                "concat_segments_indices":batch[1],
                "attention_mask":batch[2],
            }
            outputs = model(**inputs)
            targets = batch[3]
            # targets = batch['polarity'].to(args.device)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
            n_total += len(outputs)
            loss_total += loss.item() * len(outputs)

            if global_step % args.log_step == 0:
                train_acc = n_correct / n_total
                train_loss = loss_total / n_total
                logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))
        
        val_acc, val_f1 = evaluate(args, model, val_dataloader)
        logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            max_val_epoch = i_epoch
            if not os.path.exists('state_dict'):
                os.mkdir('state_dict')
            path = 'state_dict/{0}_val_acc_{1}'.format(args.dataset, round(val_acc, 4))
            torch.save(model.state_dict(), path)
            logger.info('>> saved: {}'.format(path))
        if val_f1 > max_val_f1:
            max_val_f1 = val_f1
        if i_epoch - max_val_epoch >= args.patience:
            print('>> early stop.')
            break
    return path

def evaluate(args, model, data_loader):
    n_correct, n_total = 0, 0
    targets_all, outputs_all = None, None
    # switch model to evaluation mode
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            batch = tuple(t.to(args.device) for t in batch.values())
            inputs = {
                "concat_bert_indices":batch[0],
                "concat_segments_indices":batch[1],
                "attention_mask":batch[2],
            }
            targets = batch[3]
            # t_targets = t_batch['polarity'].to(args.device)
            outputs = model(**inputs)
            n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
            n_total += len(outputs)
            if targets_all is None:
                targets_all = targets
                outputs_all = outputs
            else:
                targets_all = torch.cat((targets_all, targets), dim=0)
                outputs_all = torch.cat((outputs_all, outputs), dim=0)
    acc = n_correct / n_total
    f1 = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
    return acc, f1
    
def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='base', type=str, help='base, large, roberta')
    parser.add_argument('--dataset', default='laptop', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--lr', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=10, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=64, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--max_seq_len', default=128, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--seed', default=42, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float, help='set ratio between 0 and 1 for validation support')
    parser.add_argument('--topk', default=1, type=int)
    parser.add_argument('--method', default='bm25', type=str,help='bm25, tfidf, dense')
    args = parser.parse_args()

    

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(args.seed)

    dataset_files = {
        'twitter': {
            'train': './datasets/acl-14-short-data/train.raw',
            'test': './datasets/acl-14-short-data/test.raw'
        },
        'restaurant': {
            'train': './datasets/semeval14/Restaurants_Train.xml.seg',
            'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
        },
        'laptop': {
            'train': './datasets/semeval14/Laptops_Train.xml.seg',
            'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
        }
    }

    bert_files = {
        'base': './bert/bert-base-uncased',
        'large': './bert/bert-large-uncased',
        'roberta': './bert/roberta-large',
    }

    args.pretrained_bert_name = bert_files[args.model_name]

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert = AutoModel.from_pretrained(args.pretrained_bert_name)

    model = RE_BERT(bert,args)
    # model = DataParallel(model, device_ids=[0,1,2,3]).cuda()
    model.to(args.device)
    tokenizer = Tokenizer4Bert(args.max_seq_len, args.pretrained_bert_name)

    args.dataset_file = dataset_files[args.dataset]

    metric = RetriMetric(args.dataset_file['train'], word_tokenize, stops)

    trainset = ReDataset(args, 'train', tokenizer, metric, args.method) 
    testset = ReDataset(args, 'test', tokenizer, metric, args.method) 
    
    assert 0 <= args.valset_ratio < 1
    if args.valset_ratio > 0:
        valset_len = int(len(trainset) * args.valset_ratio)
        trainset, valset = random_split(trainset, (len(trainset)-valset_len, valset_len))
    else:
        valset = testset

    logger.info('size of trainset:{}'.format(len(trainset)))
    logger.info('size of valset:{}'.format(len(valset)))
    logger.info('size of testset:{}'.format(len(testset)))

    criterion = nn.CrossEntropyLoss()
    _params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(_params, lr= args.lr, weight_decay=args.l2reg)

    train_dataloader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False)
    val_dataloader = DataLoader(dataset=valset, batch_size=args.batch_size, shuffle=False)

    # reset_params(model)
    best_model_path = train(args, model, criterion, optimizer, train_dataloader, val_dataloader)
    model.load_state_dict(torch.load(best_model_path))
    test_acc, test_f1 = evaluate(args, model, test_dataloader)

    result = 'dataset:{}, test_acc: {:.2f}, test_f1: {:.2f}, topk:{}, method:{}, max_len:{}'.format(args.dataset, test_acc * 100 , test_f1 * 100, args.topk , args.method, args.max_seq_len)
    logger.info(result)
    with open('./result.txt','a') as f:
        f.write(result + '\n')


if __name__ == '__main__':
    main()
