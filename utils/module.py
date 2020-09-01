"""
@Author		:           Lee, Qin
@StartTime	:           2018/08/13
@Filename	:           module.py
@Software	:           Pycharm
@Framework  :           Pytorch
@LastModify	:           2019/05/07
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

from utils.process import Evaluator


class ModelManager(nn.Module):

    def __init__(self, args, num_word, num_slot, num_intent, num_kb, num_history, mem_sentence_size):
        super(ModelManager, self).__init__()
        # todo attention! 这个要跟随alphabet变而变的
        self.pad_idx = 0

        self.__num_word = num_word

        self.__num_slot = num_slot
        # self.__max_num_slot = max(num_slot0, num_slot1, num_slot2)
        # self.__sum_num = num_slot0 + num_slot1 + num_slot2
        self.__num_intent = num_intent
        self.__num_kb = num_kb
        self.__args = args

        self.__mem_embedding_dim = self.__args.mem_embedding_dim
        self.__add_mem = self.__args.use_mem

        self.__kb_num_vocab = num_word + num_intent+num_kb
        self.__his_num_vocab = num_word + num_intent + num_kb + num_history

        # Initialize an embedding object.
        self.__embedding = EmbeddingCollection(self.__his_num_vocab,self.__args.word_embedding_dim)
        self.__history_lstm = nn.LSTM(self.__args.word_embedding_dim, self.__args.history_hidden_dim, batch_first=True, bidirectional=True)
        self.__drop_out_layer = nn.Dropout(self.__args.dropout_rate)

        self.__history_bigru = nn.GRU(self.__args.word_embedding_dim, self.__args.history_hidden_dim, batch_first=True,bidirectional=True)


        # Initialize an LSTM Encoder object.
        self.__encoder = LSTMEncoder(
            self.__args.word_embedding_dim, #64
            self.__args.encoder_hidden_dim, #256
            self.__args.dropout_rate
        )

        # input_dim, hidden_dim, output_dim, dropout_rate
        # Initialize an self-attention layer.
        self.__attention = SelfAttention(
            self.__args.word_embedding_dim, #64
            self.__args.attention_hidden_dim, #1034
            self.__args.attention_output_dim, #128
            self.__args.dropout_rate
        )

        self._ctrnn = ContextRNN(
            input_size=self.__kb_num_vocab,
            hidden_size=self.__args.ctrnn_embedding_dim, #256
            dropout= self.__args.dropout_rate #0.4
        )

        self.__kb_mem = MemN2N(
            num_vocab=self.__kb_num_vocab,
            embedding_dim=self.__mem_embedding_dim, #256
            sentence_size=mem_sentence_size,
            max_hops=self.__args.max_hops, #6
            uc=self.__args.use_cuda
        )

        self.__his_mem = MemN2N(
            num_vocab=self.__his_num_vocab,
            embedding_dim=self.__mem_embedding_dim,  # 256
            sentence_size=mem_sentence_size,
            max_hops=self.__args.max_hops,  # 6
            uc=self.__args.use_cuda
        )

        # Initialize an Decoder object for intent.
        self.__intent_decoder = LSTMDecoder(
            input_dim= self.__args.word_embedding_dim,
            # input_dim=self.__args.word_embedding_dim ,
            hidden_dim=self.__args.intent_decoder_hidden_dim,
            output_dim=self.__num_intent,
            dropout_rate=self.__args.dropout_rate,
            embedding_dim=self.__args.intent_embedding_dim,
            mem_dim= self.__mem_embedding_dim,
            lk_dim =self.__args.local_knowledge_hidden_dim,
            his_dim = self.__args.history_hidden_dim,
            add_mem= self.__add_mem,
            att_dim = self.__args.attention_output_dim
        )
        # Initialize an Decoder object for slot.
        self.__slot_decoder0 = LSTMDecoder(
            # input_dim= self.__args.word_embedding_dim ,
            input_dim=self.__args.word_embedding_dim,
            hidden_dim=self.__args.slot_decoder_hidden_dim,
            output_dim=self.__num_slot,
            dropout_rate=self.__args.dropout_rate,
            embedding_dim=self.__args.slot_embedding_dim,
            extra_dim=self.__num_intent,
            mem_dim=self.__mem_embedding_dim,
            lk_dim=self.__args.local_knowledge_hidden_dim,
            his_dim=self.__args.history_hidden_dim,
            add_mem=self.__add_mem,
            att_dim=self.__args.attention_output_dim

        )

        self.__slot_decoder1 = LSTMDecoder(
            # input_dim= self.__args.word_embedding_dim ,
            input_dim=self.__args.word_embedding_dim,
            hidden_dim=self.__args.slot_decoder_hidden_dim,
            output_dim=self.__num_slot,
            dropout_rate=self.__args.dropout_rate,
            embedding_dim=self.__args.slot_embedding_dim,
            extra_dim=self.__num_intent,
            mem_dim=self.__mem_embedding_dim,
            lk_dim=self.__args.local_knowledge_hidden_dim,
            his_dim=self.__args.history_hidden_dim,
            add_mem=self.__add_mem,
            att_dim=self.__args.attention_output_dim

        )
        self.__slot_decoder2 = LSTMDecoder(
            # input_dim= self.__args.word_embedding_dim ,
            input_dim=self.__args.word_embedding_dim,
            hidden_dim=self.__args.slot_decoder_hidden_dim,
            output_dim=self.__num_slot,
            dropout_rate=self.__args.dropout_rate,
            embedding_dim=self.__args.slot_embedding_dim,
            extra_dim=self.__num_intent,
            mem_dim=self.__mem_embedding_dim,
            lk_dim=self.__args.local_knowledge_hidden_dim,
            his_dim=self.__args.history_hidden_dim,
            add_mem=self.__add_mem,
            att_dim=self.__args.attention_output_dim

        )

        self.__slot_module_list = nn.ModuleList([self.__slot_decoder0, self.__slot_decoder1, self.__slot_decoder2])

        # One-hot encoding for augment data feed.
        self.__intent_embedding = nn.Embedding(
            self.__num_intent, self.__num_intent
        )
        self.__intent_embedding.weight.data = torch.eye(self.__num_intent)
        self.__intent_embedding.weight.requires_grad = False

        self.__lk_slot_linear = nn.Linear(self.__num_slot,self.__args.local_knowledge_hidden_dim)
        self._lk_intent_linear = nn.Linear(self.__num_intent,self.__args.local_knowledge_hidden_dim)

        self.__his_attention_layer = QKVAttention(
            query_dim= self.__args.encoder_hidden_dim,
            key_dim= self.__args.encoder_hidden_dim,# 双向lstm的输出
            value_dim=self.__args.encoder_hidden_dim,
            hidden_dim= self.__args.attention_hidden_dim,
            output_dim=self.__args.history_hidden_dim,
            dropout_rate=self.__args.dropout_rate
        )


        self.__lk_attention_layer = QKVAttention(
            query_dim=self.__args.encoder_hidden_dim,
            key_dim=self.__args.slot_decoder_hidden_dim+self.__args.intent_decoder_hidden_dim,
            value_dim=self.__args.slot_decoder_hidden_dim + self.__args.intent_decoder_hidden_dim,
            hidden_dim=self.__args.attention_hidden_dim,
            output_dim=self.__args.local_knowledge_hidden_dim,
            dropout_rate=self.__args.dropout_rate
        )
        # self.__linear_list = nn.ModuleList()
        self.__mix_linear_layer = nn.Linear(self.__args.encoder_hidden_dim+self.__args.history_hidden_dim,self.__args.history_hidden_dim)   #lstm_hiddens
        # self.__linear_list.append(nn.Linear(self.__args.attention_output_dim,self.__args.attention_output_dim))
        # self.__linear_list.append(nn.Linear(self.__mem_embedding_dim,self.__mem_embedding_dim))  #kb_hiddens 256
        # self.__linear_list.append(nn.Linear(self.__mem_embedding_dim,self.__mem_embedding_dim))  #his_hiddens 256
        # self.__linear_list.append(nn.Linear(
        #     self.__args.local_knowledge_hidden_dim*2, self.__args.local_knowledge_hidden_dim*2))  #lk_lstm_hiddens 256

        # lstm_hiddens 256       attention_hiddens 128       kb_hiddens 256      his_hiddens 256  lk_lstm_hiddens 256

        # self.intent_init_tensor = torch.zeros(1,self.__args.local_knowledge_hidden_dim)
        # self.intent_init_tensor = nn.Parameter(
        #         torch.randn(1,self.__args.local_knowledge_hidden_dim),
        #         requires_grad=True
        #     )
        # self.slot_init_tensor = nn.Parameter(
        #         torch.randn(1,self.__args.local_knowledge_hidden_dim),
        #         requires_grad=True
        #     )


    def show_summary(self):
        """
        print the abstract of the defined model.
        """

        print('Model parameters are listed as follows:\n')

        print('\tnumber of word:                            {};'.format(self.__num_word))
        print('\tnumber of slot:                            {};'.format(self.__num_slot))
        print('\tnumber of intent:						    {};'.format(self.__num_intent))
        print('\tword embedding dimension:				    {};'.format(self.__args.word_embedding_dim))
        print('\tencoder hidden dimension:				    {};'.format(self.__args.encoder_hidden_dim))
        print('\tdimension of intent embedding:		    	{};'.format(self.__args.intent_embedding_dim))
        print('\tdimension of slot embedding:			    {};'.format(self.__args.slot_embedding_dim))
        print('\tdimension of slot decoder hidden:  	    {};'.format(self.__args.slot_decoder_hidden_dim))
        print('\tdimension of intent decoder hidden:        {};'.format(self.__args.intent_decoder_hidden_dim))
        print('\thidden dimension of self-attention:        {};'.format(self.__args.attention_hidden_dim))
        print('\toutput dimension of self-attention:        {};'.format(self.__args.attention_output_dim))
        print('\tlocal knowledge hiddens dim:               {};'.format(self.__args.local_knowledge_hidden_dim))
        print('\tmemory network hiddens dim:                {};'.format(self.__args.mem_embedding_dim))
        print('\tcontext rnn embedding dim:                 {};'.format(self.__args.ctrnn_embedding_dim))
        print('\tif use mem:                                {};'.format(self.__args.use_mem))

        print('\nEnd of parameters show. Now training begins.\n\n')

    def forward(self, text, seq_lens, text_triple=None, kb=None, dial_id=None, turn_id=None, history=None, n_predicts=None, forced_slot=None, forced_intent=None, local=None, turn_index=None, if_train=None, prev_slot=None, prev_intent=None):

        def extend_mem(mem_tmp_hiddens, seq_len):
            mem_hiddens = torch.empty_like(mem_tmp_hiddens[0].unsqueeze(0))
            for index, length in enumerate(seq_lens):
                if (index == 0):  # 初始化
                    mem_hiddens = mem_tmp_hiddens[0].unsqueeze(0)
                    for times in range(length - 1):
                        mem_hiddens = torch.cat((mem_hiddens, mem_tmp_hiddens[0].unsqueeze(0)), 0)
                    continue
                for times in range(length):  # 后面的迭代
                    mem_hiddens = torch.cat((mem_hiddens, mem_tmp_hiddens[index].unsqueeze(0)), 0)

            return mem_hiddens

        # --------------original code ---------------
        word_tensor, _ = self.__embedding(text)  # 基本操作是nn.Embedding #word_tensor19 17 256    test.size:19*17
        lstm_hiddens = self.__encoder(word_tensor, seq_lens) #130*256
        # transformer_hiddens = self.__transformer(pos_tensor, seq_lens)
        attention_hiddens = self.__attention(word_tensor, seq_lens)#130*128
        word_hidden = torch.cat([word_tensor[i][:seq_lens[i], :] for i in range(0, len(seq_lens))], dim=0)


        if (self.__add_mem):
            " -----------------memory-------------------"
            # _, ctrnn_hiddens = self._ctrnn(text, seq_lens)  # ctrnn 1*19*256
            # kb_tmp_hiddens = self.__kb_mem.load_mem(story=kb, hidden=ctrnn_hiddens, seq_len=seq_lens)
            # # kb=14*40*3 mem_tmp_hiddens=14*256
            # kb_hiddens = extend_mem(kb_tmp_hiddens, seq_lens)
            # # print(seq_lens) #[5, 6, 11, 17, 8, 3]  [14, 4, 16, 1, 3, 7]  [6, 1, 2, 1, 1, 1]

            "-----------------SDEN history-------------------------"
            H = []
            history = history.squeeze(dim=0)
            mask = (history != self.pad_idx)
            history_len = mask.sum(-1).long()
            history_len = [history_len[i] if history_len[i]>0 else 1 for i in range(len(history_len))]
            history_emb,_  = self.__embedding(history)
            history_hidden = self.__encoder(history_emb, history_len)

            att_his_hidden = self.__his_attention_layer(lstm_hiddens, history_hidden, history_hidden) #传这个

            mix_his_hidden = torch.cat([lstm_hiddens,att_his_hidden],dim=1)
            # mix_his_hidden = self.__mix_linear_layer(mix_his_hidden)
            mix_his_hidden = self.__drop_out_layer(mix_his_hidden)

            "-----------------mem history-----------------"
            # print(lstm_hiddens.size())
            # print(history_hidden.size())
            # print("__________________")
            # his_mem_hiddens = self.__his_mem(story=history, hidden=lstm_hiddens, seq_len=seq_lens)

            # for h in history:
            #     mask = (h!=self.pad_idx)
            #     length = mask.sum(1).long()
            #
            #     h_concat = torch.cat([h[i][:length[i]] for i in range(0,len(length)) ], dim=0)
            #     if h_concat.size(0)==0:
            #         h_concat = torch.zeros(1,dtype=torch.int64).cuda()
            #     # if if_train:
            #     #     h_concat = h_concat.cuda()
            #     # h_concat = h_concat.unsqueeze(0)
            #     H.append(h_concat)
            # history_len_list = [h.size(0) if h.size(0)>0  else 1 for h in H]
            # max_history_len = max(history_len_list)
            # pad_h = []
            # for h_index in range(len(H)):
            #     padding = torch.zeros(max_history_len - history_len_list[h_index],dtype=torch.int64).cuda()
            #     # if if_train:
            #     #     padding = padding.cuda()
            #     tmp_pad_h = torch.cat([H[h_index],padding] ,dim=0)
            #     tmp_pad_h = tmp_pad_h.unsqueeze(0)
            #     h_embeds, _ = self.__embedding(tmp_pad_h)
            #     h_embeds = self.__drop_out_layer(h_embeds)
            #     # h_hidden, _ = self.__history_lstm(h_embeds)
            #     pad_h.append(h_embeds)
            # raw_history_hidden = torch.cat(pad_h)
            #
            # # main action
            # encoded_history_hidden = self.__encoder(raw_history_hidden,history_len_list)
            #
            # att_his = []
            # for i,[text_len,history_len] in enumerate(zip(seq_lens,history_len_list)):
            #     word_begin_index = 0
            #     word_end_index = word_begin_index + text_len
            #     tmp_text_hidden = lstm_hiddens[word_begin_index:word_end_index,:]
            #     word_begin_index = word_end_index
            #
            #     history_begin_index = 0
            #     history_end_index = history_begin_index + history_len
            #     tmp_his_hidden = encoded_history_hidden[history_begin_index:history_end_index,:]
            #     history_begin_index = history_end_index
            #
            #     att_his_hidden = self.__his_attention_layer(tmp_text_hidden, tmp_his_hidden, tmp_his_hidden)
            #     # print(tmp_text_hidden.size())
            #     # print(tmp_his_hidden.size())
            #     att_his.append(att_his_hidden)
            # his_hidden = torch.cat(att_his,dim=0)
            # his_hidden = self.__drop_out_layer(his_hidden)
            #
            # # his_tmp_hiddens = self.__his_mem.load_mem(story=history, hidden=ctrnn_hiddens)
            # # his_hiddens = extend_mem(his_tmp_hiddens, seq_lens)
            "-----------------new local knowledge-----------------------"
            # local = local.view(local.size(0),-1)
            # # print(local)
            # encoded_local_hidden, _ = self.__embedding(local)
            # # att_local_hidden = self.__lk_attention_layer(local, lstm_hiddens, lstm_hiddens)
            # att_local = []
            # for i, text_len in enumerate(seq_lens):
            #     word_begin_index = 0
            #     word_end_index = word_begin_index + text_len
            #     tmp_text_hidden = lstm_hiddens[word_begin_index:word_end_index,:]
            #     word_begin_index = word_end_index
            #
            #     tmp_local_hidden = encoded_local_hidden[i,:]
            #     att_local_hidden = self.__lk_attention_layer(tmp_text_hidden, tmp_local_hidden, tmp_local_hidden)
            #     att_local.append(att_local_hidden)
            # local_hidden = torch.cat(att_local)
            # local_hidden = self.__drop_out_layer(local_hidden)
            "-----------------single local knowledge--------------------"#TODO
            # prev_slot = Variable(prev_slot_list[-1][0], requires_grad=True)
            # prev_intent = Variable(prev_intent_list[-1][0], requires_grad=True)
            # prev_sep = torch.zeros(prev_slot.size(0),1)
            prev_slot = prev_slot.squeeze(1)
            prev_intent = prev_intent.squeeze(1)
            prev_knowledge = torch.cat([prev_slot, prev_intent], dim=1)
            lk_lstm_hiddens = self.__lk_attention_layer(lstm_hiddens, prev_knowledge, prev_knowledge)

            lk_lstm_hiddens = self.__drop_out_layer(lk_lstm_hiddens)
            "-----------------multiple local knowledge----------------------------"
            # local_init_tensor = nn.Parameter(torch.randn(1,prev_slot_list[0][0].size(1)+prev_intent_list[0][0].size(1)),requires_grad =True)
            #
            # all_prev_knowledge = local_init_tensor
            # for single_prev_slot, single_prev_intent in zip(prev_slot_list,prev_intent_list):
            #     # print(single_prev_slot[0].size())
            #     prev_slot = Variable(single_prev_slot[0], requires_grad=True)
            #     prev_intent = Variable(single_prev_intent[0], requires_grad=True)
            #     # prev_sep = torch.zeros(prev_slot.size(0), 1)
            #     prev_knowledge = torch.cat([prev_slot, prev_intent], dim=1)
            #
            #     sep_knowledge = local_init_tensor  #为了隔离每个轮的local
            #     all_prev_knowledge = torch.cat([all_prev_knowledge, prev_knowledge, sep_knowledge],dim=0)
            #
            # # prev_knowledge = torch.cat([prev_slot, prev_intent], dim=1)
            # lk_lstm_hiddens = self.__lk_attention_layer(word_hidden, all_prev_knowledge, all_prev_knowledge)
            "-----------------add fully connect------------------------------------ "
            # lstm_hiddens 256       attention_hiddens 128       kb_hiddens 32      his_hiddens 32  lk_lstm_hiddens 16*2
            # lstm_hiddens = self.__linear_list[0](lstm_hiddens)
            # attention_hiddens = self.__linear_list[1](attention_hiddens)
            # kb_hiddens = self.__linear_list[2](kb_hiddens)
            # his_hiddens = self.__linear_list[3](his_hiddens)
            # lk_lstm_hiddens = self.__linear_list[4](lk_lstm_hiddens)

            # hiddens = torch.cat([lstm_hiddens, attention_hiddens, kb_hiddens, his_hiddens, lk_lstm_hiddens], dim=1)
            # hiddens = torch.cat([lstm_hiddens, mix_his_hidden],dim=1)
            # hiddens = torch.cat([lstm_hiddens, lk_lstm_hiddens],dim=1) #不加mem
            hiddens = mix_his_hidden
        else:
            hiddens = torch.cat([lstm_hiddens ,attention_hiddens],dim=1) #不加mem

        pred_intent, intent_out = self.__intent_decoder(encoded_hiddens=hiddens,seq_lens=seq_lens,forced_input=forced_intent,if_print=False)

        if not self.__args.differentiable:
            _, idx_intent = pred_intent.topk(1, dim=-1)
            feed_intent = self.__intent_embedding(idx_intent.squeeze(1))
        else:
            feed_intent = pred_intent

        _, intent_idx = pred_intent.topk(1, dim=1)
        list_intent = intent_idx.cpu().data.numpy().tolist()
        nested_intent = Evaluator.nested_list([list(Evaluator.expand_list(list_intent))], seq_lens)[0]
        option_intent = Evaluator.max_freq_predict(nested_intent)[0]

        if forced_intent is not None:
            option_intent = forced_intent[0].item()

        curr_slot_decoder = self.__slot_module_list[option_intent]
        # list_intent = intent_idx.cpu().data.numpy().tolist()
        # nested_intent = Evaluator.nested_list([list(Evaluator.expand_list(list_intent))], seq_lens)[0]
        # option_intent = Evaluator.max_freq_predict(nested_intent)[0]
        #
        # if forced_intent is not None:
        #     option_intent = forced_intent[0].item()

        # choose whhich deocder by intent
        # # curr_slot_decoder = self.__slot_module_list[option_intent]
        # if(option_intent==0):
        #     curr_slot_decoder = self.__slot_decoder0
        # elif(option_intent==1):
        #     curr_slot_decoder = self.__slot_decoder1
        # else:
        #     curr_slot_decoder = self.__slot_decoder2

        #  dim : 384 & num_intent

        # hiddens = torch.cat([lstm_hiddens, shallow], dim=1)
        pred_slot, slot_out = curr_slot_decoder(encoded_hiddens=hiddens,seq_lens=seq_lens,forced_input=forced_slot,extra_input = feed_intent,history = att_his_hidden,local = local,if_print =True)

        if if_train:
            intent_out = intent_out.squeeze(0)
            slot_out = slot_out.squeeze(0)
        else:
            intent_out = intent_out.squeeze(1)
            slot_out = slot_out.squeeze(1)

        if n_predicts is None:
            return F.log_softmax(pred_slot, dim=1), F.log_softmax(pred_intent, dim=1), slot_out, intent_out
        else:
            _, slot_index = pred_slot.topk(n_predicts, dim=1)
            _, intent_index = pred_intent.topk(n_predicts, dim=1)
            # print(pred_intent.topk(n_predicts, dim=1))
            return slot_index.cpu().data.numpy().tolist(), intent_index.cpu().data.numpy().tolist(),  slot_out, intent_out

    def golden_intent_predict_slot(self, text, seq_lens, golden_intent, n_predicts=1):
        word_tensor, _ = self.__embedding(text)
        embed_intent = self.__intent_embedding(golden_intent)

        lstm_hiddens = self.__encoder(word_tensor, seq_lens)
        attention_hiddens = self.__attention(word_tensor, seq_lens)
        hiddens = torch.cat([attention_hiddens, lstm_hiddens], dim=1)

        pred_slot = self.__slot_decoder(
            hiddens, seq_lens, extra_input=embed_intent
        )
        _, slot_index = pred_slot.topk(n_predicts, dim=-1)

        # Just predict single slot value.
        return slot_index.cpu().data.numpy().tolist()


class EmbeddingCollection(nn.Module):
    """
    Provide word vector and position vector encoding.
    """

    def __init__(self, input_dim, embedding_dim, max_len=5000):
        super(EmbeddingCollection, self).__init__()

        self.__input_dim = input_dim
        # Here embedding_dim must be an even embedding.
        self.__embedding_dim = embedding_dim
        self.__max_len = max_len

        # Word vector encoder.
        self.__embedding_layer = nn.Embedding(
            self.__input_dim, self.__embedding_dim
        )

        # Position vector encoder.
        # self.__position_layer = torch.zeros(self.__max_len, self.__embedding_dim)
        # position = torch.arange(0, self.__max_len).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, self.__embedding_dim, 2) *
        #                      (-math.log(10000.0) / self.__embedding_dim))

        # Sine wave curve design.
        # self.__position_layer[:, 0::2] = torch.sin(position * div_term)
        # self.__position_layer[:, 1::2] = torch.cos(position * div_term)
        #
        # self.__position_layer = self.__position_layer.unsqueeze(0)
        # self.register_buffer('pe', self.__position_layer)

    def forward(self, input_x):
        # Get word vector encoding.
        embedding_x = self.__embedding_layer(input_x)

        # Get position encoding.
        # position_x = Variable(self.pe[:, :input_x.size(1)], requires_grad=False)

        # Board-casting principle.
        return embedding_x, embedding_x


class LSTMEncoder(nn.Module):
    """
    Encoder structure based on bidirectional LSTM.
    """

    def __init__(self, embedding_dim, hidden_dim, dropout_rate):
        super(LSTMEncoder, self).__init__()

        # Parameter recording.
        self.__embedding_dim = embedding_dim
        self.__hidden_dim = hidden_dim // 2
        self.__dropout_rate = dropout_rate

        # Network attributes.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(
            input_size=self.__embedding_dim,
            hidden_size=self.__hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=self.__dropout_rate,
            num_layers=1
        )

    def forward(self, embedded_text, seq_lens):
        """ Forward process for LSTM Encoder.

        (batch_size, max_sent_len)
        -> (batch_size, max_sent_len, word_dim)
        -> (batch_size, max_sent_len, hidden_dim)
        -> (total_word_num, hidden_dim)

        :param embedded_text: padded and embedded input text.
        :param seq_lens: is the length of original input text.
        :return: is encoded word hidden vectors.
        """

        # Padded_text should be instance of LongTensor.
        dropout_text = self.__dropout_layer(embedded_text)

        # Pack and Pad process for input of variable length.
        packed_text = pack_padded_sequence(dropout_text, seq_lens, batch_first=True, enforce_sorted=False) #将一个填充后的变长序列压紧
        lstm_hiddens, (h_last, c_last) = self.__lstm_layer(packed_text) #经过nn.LSTM将补全padding之后的packed_text编码成lstm隐状态
        padded_hiddens, _ = pad_packed_sequence(lstm_hiddens, batch_first=True)
        # return padded_hiddens
        res = torch.cat([padded_hiddens[i][:seq_lens[i], :] for i in range(0, len(seq_lens))], dim=0)

        return res


class LSTMDecoder(nn.Module):
    """
    Decoder structure based on unidirectional LSTM.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate, mem_dim, lk_dim, his_dim, add_mem=False, embedding_dim=None, extra_dim=None, max_output_dim = None, att_dim =None):
        """ Construction function for Decoder.

        :param input_dim: input dimension of Decoder. In fact, it's encoder hidden size.
        :param hidden_dim: hidden dimension of iterative LSTM.
        :param output_dim: output dimension of Decoder. In fact, it's total number of intent or slot.
        :param dropout_rate: dropout rate of network which is only useful for embedding.
        :param embedding_dim: if it's not None, the input and output are relevant.
        :param extra_dim: if it's not None, the decoder receives information tensors.
        """

        super(LSTMDecoder, self).__init__()
        if(add_mem):
            self.__input_dim = input_dim +his_dim #第三个mem_dim其实是history_dim
        else:
            self.__input_dim = input_dim

        self.__hidden_dim = hidden_dim   #intent decoder dim 8 or slot_decoder_dim 64
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate
        self.__embedding_dim = embedding_dim
        self.__extra_dim = extra_dim

        self.__his_dim = his_dim
        # If embedding_dim is not None, the output and input
        # of this structure is relevant.
        if self.__embedding_dim is not None:
            self.__embedding_layer = nn.Embedding(output_dim, embedding_dim)
            self.__init_tensor = nn.Parameter(
                torch.randn(1, self.__embedding_dim),
                requires_grad=True
            )

        # Make sure the input dimension of iterative LSTM.
        if self.__extra_dim is not None and self.__embedding_dim is not None:
            lstm_input_dim = self.__input_dim + self.__extra_dim + self.__embedding_dim
        elif self.__extra_dim is not None:
            lstm_input_dim = self.__input_dim + self.__extra_dim
        elif self.__embedding_dim is not None:
            lstm_input_dim = self.__input_dim + self.__embedding_dim
        else:
            lstm_input_dim = self.__input_dim

        # Network parameter definition.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(
            input_size=lstm_input_dim + his_dim,
            hidden_size=self.__hidden_dim,
            batch_first=True,
            bidirectional=False,
            dropout=self.__dropout_rate,
            num_layers=1
        )

        self.__lstm_layer2 = nn.LSTM(
            input_size=self.__hidden_dim,
            hidden_size=self.__hidden_dim,
            batch_first=True,
            bidirectional=False,
            dropout=self.__dropout_rate,
            num_layers=1
        )

        self.__linear_layer = nn.Linear(
            self.__hidden_dim,
            self.__output_dim
        )
        self.__H_linear_layer = nn.Linear(
            lstm_input_dim+his_dim,
            lstm_input_dim
        )

        self.__H_att_layer = QKVAttention(
            query_dim= lstm_input_dim + his_dim,
            key_dim= lstm_input_dim + his_dim,
            value_dim= lstm_input_dim + his_dim,
            hidden_dim= hidden_dim,
            output_dim= lstm_input_dim,
            dropout_rate=self.__dropout_rate
        )

        # if extra_dim is not None:
        #     print(input_dim+input_dim+his_dim+extra_dim)  578
        if extra_dim is None:
            ed = 0
        else:
            ed = extra_dim
        self.__session_encoder = nn.GRU(self.__input_dim+his_dim+ed, his_dim, batch_first=True, bidirectional= False)



    def forward(self, encoded_hiddens, seq_lens, forced_input=None, extra_input=None, history=None, local=None, if_print=None ):
        """
            Forward process for decoder.
            :param encoded_hiddens: is encoded hidden tensors produced by encoder.
            :param seq_lens: is a list containing lengths of sentence.
            :param forced_input: is truth values of label, provided by teacher forcing.
            :param extra_input: comes from another decoder as information tensor.
            :return: is distribution of prediction labels.
        """

        # Concatenate information tensor if possible.
        # slot-filling时，有extra_input
        if extra_input is not None:
            input_tensor = torch.cat([encoded_hiddens, extra_input], dim=1)
        else:
            # print("encoder size {}".format(encoded_hiddens.size()))
            input_tensor = encoded_hiddens
        # final return output list
        output_tensor_list, sent_start_pos = [], 0
        lstm_out_list = []
        # 在train的时候 进入这个分支
        if self.__embedding_dim is None:
            print("embedding dim is none")

        if history is not None:
            mix_his = torch.cat([history, input_tensor], dim=1)
            mix_his = mix_his.unsqueeze(0)
            _, H = self.__session_encoder(mix_his)
        else:
            H = torch.zeros(1, 1, self.__his_dim).cuda()
        C = torch.zeros(1, 1, self.__his_dim).cuda()

        new_input_list = []
        for single_o in input_tensor:
            tmp_o = single_o.unsqueeze(0)
            tmp_H = H.squeeze(0)
            new_input_list.append(torch.cat([tmp_o, tmp_H], dim=1))
        new_input = torch.cat(new_input_list, dim=0)


        if self.__embedding_dim is None or forced_input is not None:
            for sent_i in range(0, len(seq_lens)):
                sent_end_pos = sent_start_pos + seq_lens[sent_i]
                # Segment input hidden tensors.
                seg_hiddens = new_input[sent_start_pos: sent_end_pos, :]
                # ---------------end here----------------------
                if self.__embedding_dim is not None and forced_input is not None:
                    # print("confirm here") #在train里 每一次都是进这里
                    # self init is random tensor
                    # TODO Thinking why set seq_lens>1 enter here,when set 0 ,no error
                    if seq_lens[sent_i] > 0: # original is 1
                        seg_forced_input = forced_input[sent_start_pos: sent_end_pos]
                        seg_forced_tensor = self.__embedding_layer(seg_forced_input).view(seq_lens[sent_i], -1)
                        seg_prev_tensor = torch.cat([self.__init_tensor, seg_forced_tensor[:-1, :]], dim=0)
                    else:
                        seg_prev_tensor = self.__init_tensor  # only when seq_len =1 ,enter here
                    # Concatenate forced target tensor.
                    combined_input = torch.cat([seg_hiddens, seg_prev_tensor], dim=1)
                else:
                    print("where am I")  #似乎并不从这里走
                    combined_input = seg_hiddens
                # 输入数据格式：
                # input(seq_len, batch, input_size)     h0(num_layers * num_directions, batch, hidden_size)     c0(num_layers * num_directions, batch, hidden_size)
                # 输出数据格式：
                # output(seq_len, batch, hidden_size * num_directions)  hn(num_layers * num_directions, batch, hidden_size) cn(num_layers * num_directions, batch, hidden_size)

                dropout_input = self.__dropout_layer(combined_input) #256+16 or 256+64
                # new_input = self.__H_linear_layer(new_input)
                # att_new_input = self.__H_att_layer(new_input,new_input,new_input)
                lstm_out, (h,c) = self.__lstm_layer(dropout_input.view(1,seq_lens[sent_i],-1),(H,C))
                # print(h.size())  # 1 1 64
                # lstm_out2 ,(h2,c2) = self.__lstm_layer2(lstm_out,(h,c))
                linear_out = self.__linear_layer(lstm_out.view(seq_lens[sent_i], -1))
                output_tensor_list.append(linear_out)
                sent_start_pos = sent_end_pos
                lstm_out_list.append(lstm_out)

                # if history is not None:
                #     print(seg_hiddens.size())
                #     print(history.size())
                #     curr_len = seg_hiddens.size(0)+history.size(0)
                #     his_combined_input = torch.cat([seg_hiddens, history], dim=0)
                #     his_dropout_input = self.__dropout_layer(his_combined_input)
                #     his_lstm_out, _ = self.__lstm_layer(his_dropout_input.view(1, curr_len, -1))
                #     his_linear_out = self.__linear_layer(his_lstm_out.view(curr_len, -1))


        # DEV TEST Validate时 进入这个分支
        else:
            # self.__embedding_dim 非空 并且 forced_input 为空:
            for sent_i in range(0, len(seq_lens)):
                prev_tensor = self.__init_tensor
                # It's necessary to remember h and c state
                # when output prediction every single step.
                last_h, last_c = None, None
                sent_end_pos = sent_start_pos + seq_lens[sent_i]


                # seg_hidden = input_tensor[sent_start_pos: sent_end_pos, :]
                # combined_input = torch.cat([seg_hidden, prev_tensor], dim=1)
                # dropout_input = self.__dropout_layer(combined_input).view(seq_lens[sent_i], 1, -1)
                # lstm_out,_ = self.__lstm_layer(dropout_input)
                # linear_out = self.__linear_layer(lstm_out.view(seq_lens[sent_i],-1))
                # output_tensor_list.append(linear_out)
                #
                # sent_start_pos = sent_end_pos

                for word_i in range(sent_start_pos, sent_end_pos):
                    seg_input = new_input[[word_i], :]
                    combined_input = torch.cat([seg_input, prev_tensor], dim=1)
                    dropout_input = self.__dropout_layer(combined_input).view(1, 1, -1)

                    # 上面训练得到的lstm(含参数)用在这里
                    if last_h is None and last_c is None:
                        lstm_out, (last_h, last_c) = self.__lstm_layer(dropout_input)
                    else:
                        lstm_out, (last_h, last_c) = self.__lstm_layer(dropout_input, (last_h, last_c))

                    lstm_out_list.append(lstm_out)

                    lstm_out = self.__linear_layer(lstm_out.view(1, -1))
                    output_tensor_list.append(lstm_out)

                    _, index = lstm_out.topk(1, dim=1)
                    prev_tensor = self.__embedding_layer(index).view(1, -1)
                sent_start_pos = sent_end_pos

        return torch.cat(output_tensor_list, dim=0), torch.cat(lstm_out_list, dim=0)


class QKVAttention(nn.Module):
    """
    Attention mechanism based on Query-Key-Value architecture. And
    especially, when query == key == value, it's self-attention.
    """

    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, output_dim, dropout_rate):
        super(QKVAttention, self).__init__()

        # Record hyper-parameters.
        self.__query_dim = query_dim
        self.__key_dim = key_dim
        self.__value_dim = value_dim

        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Declare network structures.
        self.__query_layer = nn.Linear(self.__query_dim, self.__hidden_dim)
        self.__key_layer = nn.Linear(self.__key_dim, self.__hidden_dim)
        self.__value_layer = nn.Linear(self.__value_dim, self.__output_dim)

        self.__dropout_layer = nn.Dropout(p=self.__dropout_rate)

    def forward(self, input_query, input_key, input_value):
        """ The forward propagation of attention.

        Here we require the first dimension of input key
        and value are equal.

        :param input_query: is query tensor, (n, d_q)
        :param input_key:  is key tensor, (m, d_k)
        :param input_value:  is value tensor, (m, d_v)
        :return: attention based tensor, (n, d_h)
        """

        # Linear transform to fine-tune dimension.
        linear_query = self.__query_layer(input_query)
        linear_key = self.__key_layer(input_key)
        linear_value = self.__value_layer(input_value)
        # Q*K/根号V
        score_tensor = F.softmax(torch.matmul(
            linear_query,
            linear_key.transpose(-2, -1)
        ), dim=-1) / math.sqrt(self.__hidden_dim)
        forced_tensor = torch.matmul(score_tensor, linear_value)
        forced_tensor = self.__dropout_layer(forced_tensor)
        return forced_tensor

class SelfAttention(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(SelfAttention, self).__init__()

        # Record parameters.
        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Record network parameters.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__attention_layer = QKVAttention(
            self.__input_dim, self.__input_dim, self.__input_dim,
            self.__hidden_dim, self.__output_dim, self.__dropout_rate
        )

    def forward(self, input_x, seq_lens):
        dropout_x = self.__dropout_layer(input_x)
        attention_x = self.__attention_layer(
            dropout_x, dropout_x, dropout_x
        )

        flat_x = torch.cat(
            [attention_x[i][:seq_lens[i], :] for
             i in range(0, len(seq_lens))], dim=0
        )
        return flat_x

# ---------add here---------------------

class ContextRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, n_layers=1):
        super(ContextRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

        self.embedding = nn.Embedding(input_size, hidden_size)

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

        self.W = nn.Linear(2*hidden_size, hidden_size)

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        return torch.zeros(2, bsz, self.hidden_size)

    # size的另一个变迁：torch.Size([42, 4, 6])   torch.Size([42, 24, 128])  torch.Size([42, 4, 6, 128])  torch.Size([42, 4, 128])  torch.Size([42, 4, 128])    torch.Size([2, 4, 128])     torch.Size([2, 4, 128])     torch.Size([1, 4, 128])
    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        # print(input_seqs.size()) # 53 3 6   53是该batch最长的kb的len 3是batch size 6是MEMTOKENSIZE 19*17
        embedded = self.embedding(input_seqs.contiguous().view(input_seqs.size(0), -1).long()) #53 18 128     19*17*256
        embedded = embedded.view(input_seqs.size()+(embedded.size(-1),))# 53 3 6 128      19*17*256

        # text是否是四元组的关键点 如果text是where are you 这种 就注释掉,如果是三元组 那么就解封
        # embedded = torch.sum(embedded, 2).squeeze(2) # 53 3 128    128    19*17

        embedded = self.dropout_layer(embedded)#53 3 128     19*17*256
        embedded= embedded.transpose(0,1)#17*19*256
        hidden = self.get_state(input_seqs.size(0)) #3 #2 3 128        2*19*256

        if input_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False, enforce_sorted=False)

        outputs, hidden = self.gru(embedded, hidden)
        if input_lengths:
           outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)

        # print(hidden.size())  # 2 3 128
        # hidden是用于查询Knowledge的最终隐状态
        # print(hidden.size())#2 19 256
        hidden = self.W(torch.cat((hidden[0], hidden[1]), dim=1)).unsqueeze(0) #1 19 256
        # 写回dialogue history，更新
        outputs = self.W(outputs)
        return outputs.transpose(0,1), hidden

class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i)) #getattr() 函数用于返回一个对象属性值。 module中的属性 predix+str(i) 的值

class MemN2N(nn.Module):
    def __init__(self,num_vocab,embedding_dim,sentence_size,max_hops,uc=False):
        super(MemN2N, self).__init__()

        use_cuda = uc
        self.max_hops = max_hops

        for hop in range(self.max_hops + 1):
            C = nn.Embedding(num_vocab, embedding_dim, padding_idx=0)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        # print("num vocab{} embeding {}".format(num_vocab,embedding_dim))

    # def forward(self):
    #     return


    def forward(self, story, hidden, seq_len=None,p=None):
        # Forward multiple hop mechanism
        u = [hidden]  #hidden是1×19×256
        # 就不经过mem添加了 两个size是一样的(这里的,和结尾的)
        if story.size()[-1] == 0:
            return u[-1]
        story_size = story.size()  #19 56 3 MEM_TOKEN_SIZE

        print(story.size())
        print(hidden.size())

        # print(self.C.size())
        for hop in range(self.max_hops):
            # if p:
            #     print(story>=0)
            embed_A = self.C[hop](story.contiguous().view(story_size[0], -1))  # .long()) # b *(m * s) * e embedA # 19 168 256

            embed_A = embed_A.view(story_size + (embed_A.size(-1),))  # b * m * s * e embedA #19 56 3 256
            embed_A = torch.sum(embed_A, 1).squeeze(1)  # b * m * e # embedA 19 56 256

            # dialogue history H； add language model embedding
            # if not args["ablationH"]:
            #     embed_A = self.add_lm_embedding(embed_A, kb_len, conv_len, dh_outputs)

            if (len(list(u[-1].size())) == 1):
                u[-1] = u[-1].unsqueeze(0)  ## used for bsz = 1.

            print(embed_A.size())
            print(u[-1].size())
            u_temp = u[-1].unsqueeze(1).expand_as(embed_A)
            prob_logit = torch.sum(embed_A * u_temp, 2)
            prob_ = self.softmax(prob_logit)

            embed_C = self.C[hop + 1](story.contiguous().view(story_size[0], -1).long())
            embed_C = embed_C.view(story_size + (embed_C.size(-1),))
            embed_C = torch.sum(embed_C, 1).squeeze(1)
            # if not args["ablationH"w]:
            #     embed_C = self.add_lm_embedding(embed_C, kb_len, conv_len, dh_outputs)
            prob = prob_.unsqueeze(2).expand_as(embed_C)
            o_k = torch.sum(embed_C * prob, 1)
            u_k = u[-1] + o_k  # 更新q(k+1)=q(k)+o(k)
            u.append(u_k)
            self.m_story.append(embed_A)
        self.m_story.append(embed_C)
        return u[-1]
