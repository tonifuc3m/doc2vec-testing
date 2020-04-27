import gensim.models as g
import logging
import argparse
import os
import codecs
import sys
import numpy as np

def mkdir_p(path):
    try:
        os.makedirs(path)
    except:
        pass


def main():
    parser = argparse.ArgumentParser(description='Doc2vec driver.')
    parser.add_argument('mode', choices=['train', 'retrieve', 'infer'], help='Training, retrieve trained embeddings'
                                                                             'or inference mode')
    parser.add_argument('name', type=str, help='Model name')
    parser.add_argument('output_path', type=str, help='Output path')
    parser.add_argument('word_embeddings_path', type=str,
                        help='Pre-trained word embeddings path')
    parser.add_argument('tokenized_path', type=str,
                        help='Directory with tokenized plain text documents')
    args = parser.parse_args()


    if None in [args.mode, args.name, args.output_path]:
        exit('Arguments mode, name, output_path are required')
    if args.mode == 'train' and (args.word_embeddings_path is None or args.tokenized_path is None):
        exit('word_embeddings_path and tokenized_path arguments are required if mode is set to train')
    if args.mode == 'infer' and args.tokenized_path is None:
        exit('word_embeddings_path and tokenized_path arguments are required if mode is set to train')
    if args.mode == 'train':
        pretrained_emb = args.word_embeddings_path
        tokenized_path = args.tokenized_path
        texts = []
        for filename in os.listdir(tokenized_path):
            with codecs.open(os.path.join(tokenized_path, filename), 'r', 'utf-8') as f:
                doc_tokens = []
                for line in f.readlines():
                    if len(line) > 0:
                        doc_tokens += line.split()
                texts.append(doc_tokens)
                #texts.append([line.split() for line in f.readlines()])


        #doc2vec parameters
        vector_size = 300
        window_size = 15
        min_count = 1
        sampling_threshold = 1e-5
        negative_size = 5
        train_epoch = 5
        dm = 0 #0 = dbow; 1 = dmpv
        worker_count = 6 #number of parallel processes
        saved_path = os.path.join(args.output_path, 'models', args.name + '.bin')

        #enable logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        mkdir_p(os.path.join(args.output_path, 'models'))

        docs = [g.doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
        model = g.Doc2Vec(docs, size=vector_size, window=window_size, min_count=min_count, sample=sampling_threshold,
                          workers=worker_count, hs=0, dm=dm, negative=negative_size, dbow_words=1, dm_concat=1,
                          pretrained_emb=pretrained_emb, iter=train_epoch)
        model.save(saved_path)
    elif args.mode == 'retrieve':
        saved_path = os.path.join(args.output_path, 'models', args.name + '.bin')
        model = g.Doc2Vec.load(saved_path)
        vectors = []
        for i in range(len(model.docvecs)):
            vectors.append(model.docvecs[i])
        vectors = np.array(vectors)
        np.save(os.path.join(args.output_path, args.name + '_vectors'), vectors)
            # print model.docvecs[i]
    else: # infer
        # inference hyper - parameters
        start_alpha = 0.01
        infer_epoch = 1000

        # load model
        m = g.Doc2Vec.load(os.path.join(args.output_path, 'models', args.name + '.bin'))
        tokenized_path = args.tokenized_path
        texts = []
        for filename in os.listdir(tokenized_path):
            with codecs.open(os.path.join(tokenized_path, filename), 'r', 'utf-8') as f:
                texts.append([line.split() for line in f.readlines()])
        test_docs = texts
        # infer test vectors
        for d in test_docs:
            print ' '.join([str(x) for x in m.infer_vector(d, alpha=start_alpha, steps=infer_epoch)]) + '\n'


if __name__ == '__main__':
    main()