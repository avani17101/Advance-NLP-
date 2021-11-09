#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.layers import Layer, Dropout,Lambda, Embedding, TimeDistributed,  Dense, SpatialDropout1D, LSTM, Input,add, concatenate
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
import tensorflow as tf
import re
import tensorflow.keras.backend as K
import scipy.spatial.distance as ds
from argparse import ArgumentParser


model_path = 'data/models'


class DropoutTimestaps(Dropout):
    def __init__(self, rate, **kwargs):
        super(DropoutTimestaps, self).__init__(rate, **kwargs)

    def get_config(self):
        return super(DropoutTimestaps, self).get_config()

    def _get_noise_shape(self, inputs):
        return (K.shape(inputs)[0],K.shape(inputs)[1], 1)


class Camouflage(Layer):
    # getting places where input is not equal-to mask value(non-zero in this implementation)
    def __init__(self, mask_value=0., **kwargs):
        super(Camouflage, self).__init__(**kwargs)
        self.mask_value = mask_value

    def call(self, inputs):
        x, y = inputs
        temp = K.not_equal(y, self.mask_value)
        temp = K.any(temp, axis=-1, keepdims=True)
        temp = K.cast(temp, K.dtype(x))
        return x * temp

    def get_config(self):
        base_config = super(Camouflage, self).get_config().copy()
        base_config['mask_value'] = self.mask_value
        return base_config

    def compute_output_shape(self, input_shape):
        return input_shape[0]



class SampledAct(Layer):
    def __init__(self,  num_sampled=1000,num_classes=50000, checkpoint_wt=False, **kwargs):
        super(SampledAct, self).__init__(**kwargs)
        [self.num_classes, self.num_sampled,self.checkpoint_wt]= (num_classes,num_sampled,checkpoint_wt)

    def build(self, _):
        [self.softmax_b, self.built] = (self.add_weight(shape=(self.num_classes,), name='b_soft', initializer='zeros'), True)

    def call(self, x):        
        def sampled_act(x):
            batch_losses = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                self.checkpoint_wt.weights[0], self.softmax_b,
                x[1], x[0],
                num_classes=self.num_classes,
                num_sampled=self.num_sampled))
            return [batch_losses]*2

        self.add_loss(0.5 * tf.reduce_mean(tf.map_fn(sampled_act, x)[0][0]))
        return x[0]

    def compute_output_shape(self, input_shape):
        return input_shape[0] 

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_classes': self.num_classes,
            'checkpoint_wt': self.checkpoint_wt
        })
        return config
    

  
class ELMo(object):
    def __init__(self, params):
        [self._model,self._elmo_model,self.params] = (None,None, params)
        self.sd = SpatialDropout1D(self.params['dropout_rate'])
        self.td = DropoutTimestaps(self.params['word_dropout_rate'])
        self.embeds = Embedding(self.params['vocab_size'], self.params['hidden_units_size'], trainable=True, name='token_encoding')
        self.sampled_act = SampledAct(num_classes=self.params['vocab_size'],
                                         num_sampled=int(self.params['num_sampled']),
                                         checkpoint_wt=self.embeds)

        self.mm_norm = MinMaxNorm(-1 * params['cell_clip'],params['cell_clip'])
        self.lstm_net = LSTM(units=params['lstm_units_size'], return_sequences=True, activation="tanh",
                        recurrent_activation='sigmoid',
                        kernel_constraint= self.mm_norm,
                        recurrent_constraint= self.mm_norm)

        self.camo = Camouflage(mask_value=0)
        self.td_ = TimeDistributed(Dense(params['hidden_units_size'], activation='linear',
                                         kernel_constraint=MinMaxNorm(-1 * params['proj_clip'],
                                                                      params['proj_clip'])))

        self.sd_ = SpatialDropout1D(params['dropout_rate'])
        self.compile_elmo()

    def __del__(self):
        K.clear_session()
        del self._model
    
    def lstm_pass(self, lstm_inputs, drop_inputs, type="f"):
        for i in range(2): #2 layers of LSTMs
            lstm = self.lstm_net(lstm_inputs)
            lstm = self.camo(inputs=[lstm, drop_inputs])
            proj = self.td_(lstm)
            lstm_inputs = add([proj, lstm_inputs], name=type+'_block_{}'.format(i + 1))
            lstm_inputs = self.sd_(lstm_inputs)
        return lstm_inputs
        
    def compile_elmo(self):
        embeds = Embedding(self.params['vocab_size'], self.params['hidden_units_size'], trainable=True, name='token_encoding')

        word_inputs = Input(shape=(None,), name='word_indices', dtype='int32')
        drop_inputs = self.sd(embeds(word_inputs))
        lstm_inputs = self.td(drop_inputs)

        next_inputs = Input(shape=(None, 1), name='next_inputs', dtype='float32')
        previous_inputs = Input(shape=(None, 1), name='previous_inputs', dtype='float32')

        re_lstm_inputs = Lambda(function=ELMo.reverse)(lstm_inputs)
        mask = Lambda(function=ELMo.reverse)(drop_inputs)

        # Forward LSTMs
        lstm_inputs = self.lstm_pass(lstm_inputs, drop_inputs)


        # Backward LSTMs
        re_lstm_inputs = self.lstm_pass(lstm_inputs, mask, type='b')

        # Project to Vocabulary with Sampled Softmax
        sampled_act = SampledAct(num_classes=self.params['vocab_size'],
                                         num_sampled=int(self.params['num_sampled']),
                                         checkpoint_wt=embeds)

        outputs = sampled_act([lstm_inputs, next_inputs])
        

        re_lstm_inputs = Lambda(function=ELMo.reverse, name="reverse")(re_lstm_inputs)

        re_outputs = sampled_act([re_lstm_inputs, previous_inputs])

        self._model = Model(inputs=[word_inputs, next_inputs, previous_inputs],
                            outputs=[outputs, re_outputs])
        self._model.compile(optimizer=Adagrad(lr=self.params['lr'], clipvalue=self.params['clip_value']),
                            loss=None)
        self._model.summary()

    def train(self, train_data, valid_data):
        save_best_model = ModelCheckpoint(filepath=model_path + "/"+ "elmo_best_weights.hdf5", monitor='val_loss', verbose=1,
                                          save_best_only=True, mode='auto')
        self._model.fit_generator(train_data,
                                  validation_data=valid_data,
                                  epochs=self.params['epochs'],
                                  workers = os.cpu_count(),
                                  use_multiprocessing=False,
                                  callbacks=[save_best_model])



    def save_elmo_encoder(self):
        embed = self._model.get_layer('token_encoding').output
        elmo_embeds = [concatenate([embed, embed], name='elmo_embeds_level_0')]

        for i in range(2):
            fb = self._model.get_layer('f_block_{}'.format(i + 1)).output
            bb = self._model.get_layer('b_block_{}'.format(i + 1)).output
            elmo_embeds.append(concatenate([fb,Lambda(function=ELMo.reverse)(bb)],name='elmo_embeds_level_{}'.format(i + 1)))

        camos = []
        for i, elmo_embedding in enumerate(elmo_embeds):
            camo = Camouflage(mask_value=0.0, name='camo_elmo_embeds_level_{}'.format(i + 1))
            camos.append(camo([elmo_embedding, self._model.get_layer('token_encoding').output]))

        self._elmo_model = Model(inputs=[self._model.get_layer('word_indices').input], outputs=camos)
        self._elmo_model.summary()
        if not model_path.endswith("/"):
            checkpoint_path = model_path + "/%s"
        else:
            checkpoint_path = model_path + "%s"
        self._elmo_model.save(checkpoint_path % ('ELMo_Encoder.hd5'))

    def save(self, sampled_act=True):
        if sampled_act:
            pass
        else:
            self.params['num_sampled'] = self.params['vocab_size']
        self.compile_elmo()
        if not model_path.endswith("/"):
            checkpoint_path = model_path + "/%s"
        else:
            checkpoint_path = model_path + "%s"
        self._model.load_weights(checkpoint_path % ('elmo_best_weights.hdf5', ))
        self._model.save(checkpoint_path % ('ELMo_LM_EVAL.hd5', ))

    def load(self):
        if not model_path.endswith("/"):
            checkpoint_path = model_path + "/%s"
        else:
            checkpoint_path = model_path + "%s"
        self._model.load_weights(checkpoint_path % ('elmo_best_weights.hdf5', ))

    def load_elmo_encoder(self):
        if not model_path.endswith("/"):
            checkpoint_path = model_path + "/%s"
        else:
            checkpoint_path = model_path + "%s"

        self._elmo_model = load_model(checkpoint_path% ('ELMo_Encoder.hd5',),
                                      custom_objects={'DropoutTimestaps': DropoutTimestaps,
                                                      'Camouflage': Camouflage})

    def get_embedding(self,test_data):
        elmo_vectors_lis = []
        for i in range(len(test_data)):
            x = test_data[i]
            preds = np.asarray(self._elmo_model.predict(x))
            
            elmo_vectors = np.mean(preds, axis=0)
            elmo_vectors_lis.append(elmo_vectors)
        return np.array(elmo_vectors_lis)
        

    @staticmethod
    def reverse(inputs, axes=1):
        return K.reverse(inputs, axes=axes)


class DataLoaderLM(tf.keras.utils.Sequence):
    '''
    DataLoader for converting word tokens to ordinal encoding
    pads sentences to sentence_maxlen
    corpus: list of tokenised sentences
    vocab: vocablary of words
    params: parameters containing batch_size, sentence_maxlen
    '''
    def __len__(self):
        ind = len(self.indices)/self.params['batch_size']
        return int(np.ceil(ind))

    def __init__(self, corpus, vocab, params):
        self.corpus = corpus
        self.vocab = vocab
        self.sent_ids = corpus
        self.params = params
        self.indices = np.delete(np.arange(len(corpus)), [idx for idx in range(0, len(corpus), 2)])

    def __getitem__(self, index):
        start_ind = index * self.params['batch_size']
        end_ind = (index + 1) * self.params['batch_size']
        batch_indices = self.indices[start_ind:end_ind]
        
        forward_word_ind_batch = np.zeros((len(batch_indices), self.params['sentence_maxlen']), dtype=np.int32)
        backward_word_ind_batch = np.zeros((len(batch_indices), self.params['sentence_maxlen']), dtype=np.int32)
        word_ind_batch = np.zeros((len(batch_indices), self.params['sentence_maxlen']), dtype=np.int32)
        
        for i, batch in enumerate(batch_indices):
            word_ind_batch[i] = self.get_word_mappings(batch)

        for i, ws in enumerate(word_ind_batch):
            forward_word_ind_batch[i] = np.concatenate((ws[1:], np.zeros((1,), dtype=np.int32)), axis=0)
            backward_word_ind_batch[i] = np.concatenate((np.zeros((1,), dtype=np.int32), ws[:-1]), axis=0)

        forward_word_ind_batch = forward_word_ind_batch[:, :, np.newaxis]
        backward_word_ind_batch = backward_word_ind_batch[:, :, np.newaxis]

        return [word_ind_batch, forward_word_ind_batch, backward_word_ind_batch], []

    def on_epoch_end(self):
            np.random.shuffle(self.indices)

    def get_word_mappings(self, idx):
        tk_id_list = np.zeros((self.params['sentence_maxlen'],), dtype=np.int32)
        tk_id_list[0] = self.vocab['<bos>']

        for j, tk in enumerate(corpus[idx][:self.params['sentence_maxlen'] - 2]):
            if tk.lower() not in self.vocab:
                tk_id_list[j + 1] = self.vocab['<unk>']
            else:
                tk_id_list[j + 1] = self.vocab[tk.lower()]
                
        if tk_id_list[1]!=0:
            tk_id_list[j + 2] = self.vocab['<eos>']
        
        return tk_id_list
    

def get_corpus():
    '''
    get preprocessed corpus
    '''
    data_path = 'data/datasets/swb_ms98_transcriptions'
    seq = [x[0] for x in os.walk(data_path)]
    files_lis = []
    for s in seq:
        if len(s)==45:
            files = [f for f in os.listdir(s) if os.path.isfile(os.path.join(s, f))]
            for f in files:
                if f.endswith('trans.text'):
                    files_lis.append(os.path.join(s,f))                 
    
    corpus = []
    # preprocessing
    for file in files_lis:
        with open(file) as f:
            for line in f.readlines():
                line = re.sub("[\{\(\[].*?[\)\}\]]", "", line.lower())
                line = re.sub("\'s","",line)
                line = re.sub("\'","",line)
                line = re.sub("-","",line)
                line = re.sub("/","",line)
                line = re.sub('[^a-z ]','',line)
                temp = line.split()[3:]
                if len(temp):
                    if len(temp)>1:
                        corpus.append(temp)
    return corpus

corpus = get_corpus()

print("num sentences",len(corpus))
n = len(corpus)

# train val split- 90:10
train_s_idx = 0
train_e_idx = int(0.9*n)

val_s_idx = train_e_idx
val_e_idx = int(val_s_idx+0.1*n)

train_data = corpus[train_s_idx:train_e_idx]
val_data = corpus[val_s_idx:val_e_idx]

#loading vocab
vocab = None
import pickle
with open('data/datasets/swb_ms98_transcriptions/vocab.txt', "rb") as fp: 
    vocab = pickle.load(fp)

vocab.sort()

vocab =  dict(enumerate(vocab, 2))
vocab = dict(zip(vocab.values(), vocab.keys()))
vocab.update({'<bos>':0, '<eos>':1, '<oov>':len(vocab)-1})

params = {
    'vocab': vocab,
    'vocab_size': len(vocab)
}

parser = ArgumentParser()
parser.add_argument("--epochs", default=10, type=int, help="num epochs")
parser.add_argument("--batch_size", default=32, type=int, help="batch size")
parser.add_argument("--lr", default=0.2, type=float, help="lr")
parser.add_argument("--sentence_maxlen", default=100, type=float, help="sentence_maxlen")

parser.add_argument("--num_sampled", default=1000, type=int, help="num sampled")
parser.add_argument("--clip_value", default=1, type=int, help="clip_value")
parser.add_argument("--cell_clip", default=5, type=int, help="cell_clip")
parser.add_argument("--proj_clip", default=5, type=int, help="proj_clip")

parser.add_argument("--lstm_units_size", default=400, type=int, help="lstm_units_size")
parser.add_argument("--hidden_units_size", default=200, type=int, help="hidden_units_size")
parser.add_argument("--dropout_rate", default=0.1, type=int, help="dropout_rate")
parser.add_argument("--word_dropout_rate", default=0.05, type=int, help="word_dropout_rate")
parser.add_argument("--mode", default="encode",  help="modes: train/encode")

opt = parser.parse_args()

params.update({"epochs":opt.epochs, "batch_size": opt.batch_size,"lr":opt.lr, "sentence_maxlen":opt.sentence_maxlen,"lr":opt.lr,"sentence_maxlen":opt.sentence_maxlen,"num_sampled":opt.num_sampled,"clip_value":opt.clip_value,"cell_clip":opt.cell_clip,"proj_clip":opt.proj_clip,"lstm_units_size":opt.lstm_units_size,"hidden_units_size":opt.hidden_units_size,"dropout_rate":opt.dropout_rate,"word_dropout_rate":opt.word_dropout_rate})

train_generator = DataLoaderLM(train_data,
                                  vocab,
                                  params)

val_generator = DataLoaderLM(val_data,
                                vocab,
                                params)


elmo_model = ELMo(params)
elmo_model.compile_elmo()

if opt.mode=='train':
    elmo_model.train(train_data=train_generator, valid_data=val_generator)
    elmo_model.save(sampled_act=False)
    elmo_model.save_elmo_encoder()
else:
    elmo_model.load()
# loading the ELMo encoder weights: if mode was train: newly trained weights are loaded, else the weights in data/models are loaded
elmo_model.load_elmo_encoder()


# pass in the sentences to be encoded by ELMo
sentences = [['that','seems','to','be','the','target','choice'],
['this', 'is', 'really','a','pretty', 'nice', 'car'],
['fashion','industry','is','growing','fast','these','days'],
['you','look','very','pretty','in','this','dress'],
["last", "year","dress","price","was" ,"forty", "dollars"]]


#ordinal encoding the sentence words
tokens = []
for sentence in sentences:
    token  = []
    for word in sentence:
        if word in vocab:
            token.append(vocab[word])
        else: #handling oov words
            token.append(vocab['<oov>'])
    tokens.append(token)
    
             
elmo_embeds = elmo_model.get_embedding(tokens)
elmo_embeds = elmo_embeds.reshape((elmo_embeds.shape[0],elmo_embeds.shape[1],elmo_embeds.shape[3]))



def compute_dist(pos1, pos2,pos3):
    '''
    computing distance between words in elmo embeddings
    '''
    e1 = np.linalg.norm(elmo_embeds[pos1[0],pos1[1],:]-elmo_embeds[pos2[0],pos2[1],:])
    e2 = np.linalg.norm(elmo_embeds[pos1[0],pos1[1],:]-elmo_embeds[pos3[0],pos3[1],:])

    c1 = ds.cosine(elmo_embeds[pos1[0],pos1[1],:],elmo_embeds[pos2[0],pos2[1],:])
    c2 = ds.cosine(elmo_embeds[pos1[0],pos1[1],:],elmo_embeds[pos3[0],pos3[1],:])

    print("\n {}-{}: ed: {} cs: {}".format(sentences[pos1[0]][pos1[1]],sentences[pos2[0]][pos2[1]],np.round(e1,2),np.round(c1,2)))
    print("\n {}-{}: ed: {} cs: {}".format(sentences[pos1[0]][pos1[1]],sentences[pos3[0]][pos3[1]],np.round(e2,2),np.round(c2,2)))


pos1 = [1,4] #word location: sentence number, word position in setence
pos2 = [0,-2]
pos3 = [1,-2]
compute_dist(pos1, pos2,pos3)


pos1 = [1,4]
pos2 = [2,-1]
pos3 = [2,3]
compute_dist(pos1, pos2,pos3)


pos1 = [1,4]
pos2 = [3,-1]
pos3 = [2,3]
compute_dist(pos1, pos2,pos3)

pos1 = [2,1]
pos2 = [3,-1]
pos3 = [2,3]
compute_dist(pos1, pos2,pos3)

pos1 = [-1,0]
pos2 = [3,-1]
pos3 = [2,3]
compute_dist(pos1, pos2,pos3)

pos1 = [0,-2] 
pos2 = [2,-3]
pos3 = [1,-2]
compute_dist(pos1, pos2,pos3)

pos1 = [0,1] 
pos2 = [-1,-1]
pos3 = [2,2]
compute_dist(pos1, pos2,pos3)

pos1 = [0,1] 
pos2 = [2,-1]
pos3 = [-1,1]
compute_dist(pos1, pos2,pos3)

pos1 = [0,1] 
pos2 = [2,-1]
pos3 = [-1,1]
compute_dist(pos1, pos2,pos3)
