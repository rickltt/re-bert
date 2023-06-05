from transformers import BertModel, BertTokenizer
import numpy as np
import os
model_dir = '/home/tongtao.ling/ltt_code/bert/bert-base-uncased'
bert = BertModel.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)

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


def retrival_corpus(dataset):
    fname = dataset_files[dataset]['train']
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    raw_data = []
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        text = text_left + " " + aspect + " " + text_right
        if text in raw_data:
            continue
        raw_data.append(text)
    print(len(raw_data))
    vecs = []
    for instance in raw_data:
        tokens = tokenizer.encode_plus(text=instance,return_tensors='pt')
        model_out = bert(**tokens)
        vec = model_out[0].mean(1).squeeze(0)
        vec = vec.detach().numpy()
        vecs.append(vec) 
    vecs = np.array(vecs)
    if not os.path.exists('dense'):
        os.mkdir('dense')
    np.save(os.path.join('./dense/',dataset),vecs)

def compute_vecs(dataset, mode):
    fname = dataset_files[dataset][mode]
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    raw_data = []
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        text = text_left + " " + aspect + " " + text_right
        raw_data.append(text)
    print(len(raw_data))
    vecs = []
    for instance in raw_data:
        tokens = tokenizer.encode_plus(text=instance,return_tensors='pt')
        model_out = bert(**tokens)
        vec = model_out[0].mean(1).squeeze(0)
        vec = vec.detach().numpy()
        vecs.append(vec) 

    vecs = np.array(vecs)
    if not os.path.exists('dense'):
        os.mkdir('dense')

    np.save(os.path.join('./dense/',dataset + '_' + mode),vecs)


if __name__ == '__main__':
    for dataset in dataset_files:
        print(dataset)
        retrival_corpus(dataset)
    
    for dataset,mode in dataset_files.items():
        print(dataset)
        for i in mode:
            print(i)
            compute_vecs(dataset, i)