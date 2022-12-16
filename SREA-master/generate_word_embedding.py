from tqdm import tqdm
import numpy as np
import json


def load_id_entity(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    ids = []
    entities = []
    for line in lines:
        line = line.strip().split('\t')
        ids.append(int(line[0]))
        entities.append(line[1])
    return ids, entities


def generate_word_embeddings(word_vecs, ent_names):
    ent_vec = np.zeros((len(ent_names), 300))
    for i, name in ent_names:
        k = 0
        for word in name:
            word = word.lower()
            if word in word_vecs:
                ent_vec[i] += word_vecs[word]
                k += 1
        if k:
            ent_vec[i] /= k
        else:
            ent_vec[i] = np.random.random(300) - 0.5
        ent_vec[i] = ent_vec[i] / np.linalg.norm(ent_vec[i])
    return ent_vec


if __name__ == '__main__':
    dataset = ['DBP15K', 'SRPRS']
    language = ['zh_en', 'ja_en', 'fr_en', 'en_fr', 'en_de']

    # load glove
    word_vecs = {}
    with open("./glove.6B.300d.txt", encoding='UTF-8') as f:
        for line in tqdm(f.readlines()):
            line = line.split()
            word_vecs[line[0]] = np.array([float(x) for x in line[1:]])

    for data in dataset:
        if data == 'DBP15K':
            for lang in language[:3]:
                save_path = 'data/' + data + '/' + lang + '/' + lang.split('_')[0] + '_vectorList.json'
                ent_names = data.lower() + '_' + lang + '.json'
                # load translated entity name
                ent_names = json.load(open("translated_ent_name" + '/' + ent_names, "r"))
                # generate word embeddings
                ent_vec = generate_word_embeddings(word_vecs, ent_names)
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(ent_vec.tolist(), f)
        if data == 'SRPRS':
            for lang in language[3:]:
                save_path = 'data/' + data + '/' + lang + '/' + lang.split('_')[1] + '_vectorList.json'
                ent_names = data.lower() + '_' + lang + '.json'
                # load translated entity name
                ent_names = json.load(open("translated_ent_name" + '/' + ent_names, "r"))
                # generate word embeddings
                ent_vec = generate_word_embeddings(word_vecs, ent_names)
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(ent_vec.tolist(), f)









