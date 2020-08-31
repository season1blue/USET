"""
@Author		:           Lee, Qin
@StartTime	:           2018/08/13
@Filename	:           process.py
@Software	:           Pycharm
@Framework  :           Pytorch
@LastModify	:           2019/05/07
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from copy import deepcopy

from sklearn.metrics import f1_score

import os
import time
import random
import numpy as np
from tqdm import tqdm
from collections import Counter
# Utils functions copied from Slot-gated model, origin url:
# 	https://github.com/MiuLab/SlotGated-SLU/blob/master/utils.py
from utils import miulab


class Processor(object):

    def __init__(self, dataset, model, args):
        self.__dataset = dataset
        self.__model = model
        self.__batch_size = args.batch_size
        self.__args = args

        if torch.cuda.is_available():
            time_start = time.time()
            self.__model = self.__model.cuda()

            time_con = time.time() - time_start
            print("The model has been loaded into GPU and cost {:.6f} seconds.\n".format(time_con))

        self.__criterion = nn.NLLLoss()
        self.__optimizer = optim.Adam(
            self.__model.parameters(), lr=self.__dataset.learning_rate,
            weight_decay=self.__dataset.l2_penalty
        )

        # self.prev_intent_list = [[torch.zeros(1, len(dataset.intent_alphabet))]]
        # self.prev_slot_list = [[torch.zeros(1, len(dataset.slot_alphabet))]]

        self.__max_column_len = args.max_column_len
        self.__slot_num = args.slot_num

        # self.prev_slot_list = [torch.zeros(1, len(self.__dataset.slot_alphabet))]
        # self.prev_intent_list = [torch.zeros(1, len(self.__dataset.intent_alphabet))]
        #
        # self.prev_slot_tensor = torch.randn(1,1)
        # self.prev_intent_tensor = torch.randn(1,1)

    # exk是DataLoader对象训练数据



    def train(self, train_index):

        best_dev_slot = 0.0
        best_dev_intent = 0.0
        best_dev_sent = 0.0
        dataloader = self.__dataset.batch_delivery('train')
        for epoch in range(0, self.__dataset.num_epoch):
            total_slot_loss, total_intent_loss = 0.0, 0.0
            time_start = time.time()
            self.__model.train()
            print("-------------------------TRAIN----------------------------------")
            for data_detail_batch in tqdm(dataloader, ncols=50):
                dial_id_batch, turn_id_batch, history_batch, slot_batch, text_batch, intent_batch, kb_batch, text_triple_batch, uuid_batch = [], [], [], [], [], [], [], [], []  #claim here
                slot_loss, intent_loss, dialogue_loss = 0.0, 0.0, 0.0
                turn_slot_loss_list = []
                turn_intent_loss_list = []
                turn_loss_list = []
                # 每一dilaogue都最新初始化
                prev_intent_list = [torch.zeros(1, self.__args.intent_decoder_hidden_dim).cuda()]
                prev_slot_list = [torch.zeros(1, self.__args.slot_decoder_hidden_dim).cuda()]


                digit_intent_list, text_intent_list = [], []
                for turn_index,turn in enumerate(data_detail_batch):
                    # 以turn为基本单位, 但是每个turn中含有该batch中所有的dialogue对应的turn 以这个indent写
                    dial_id_batch, turn_id_batch, history_batch, slot_batch, text_batch, intent_batch, kb_batch, text_triple_batch, uuid_batch = [], [], [], [], [], [], [], [], []  # init here

                    for t in turn:
                        if len(t) != 0:
                            dial_id_batch.append(t['dial_id'])
                            turn_id_batch.append(t['turn_id'])
                            history_batch.append(t['history'])
                            slot_batch.append(t['slot'])
                            text_batch.append(t['text'])
                            intent_batch.append(t['intent'])
                            kb_batch.append(t['kb'])
                            text_triple_batch.append(t['triple'])
                            uuid_batch.append(t['uuid'])
                        else:
                            dial_id_batch.append(0)
                            turn_id_batch.append(0)
                            history_batch.append([[0]])
                            slot_batch.append([0])
                            text_batch.append([0])
                            intent_batch.append([0])
                            kb_batch.append([[0,0,0]])
                            text_triple_batch.append([[0,0,0]])
                            uuid_batch.append('0')

                    "---------------------------local-------------------------"
                    # prev_slot_tensor = Variable(torch.cat(prev_slot_list, dim=0), requires_grad=True)
                    # prev_intent_tensor = Variable(torch.cat(prev_intent_list, dim=0), requires_grad=True)
                    prev_slot_tensor = Variable(prev_slot_list[-1], requires_grad=True)
                    prev_intent_tensor = Variable(prev_intent_list[-1], requires_grad=True)


                    # 在这里解决的kb 和triple 的对应和padding问题 ,
                    padded_text, seq_lens, sorted_dial_id, sorted_turn_id, sorted_history, sorted_slot, sorted_intent, sorted_kb, sorted_text_triple = self.__dataset.add_padding(
                        text_batch, dial_id_batch, turn_id_batch,history_batch, slot_batch, intent_batch, kb_batch, text_triple_batch, digital=True
                    )

                    # add padding既有padding 又有排序
                    sorted_intent = [item * num for item, num in zip(sorted_intent, seq_lens)]
                    sorted_intent = list(Evaluator.expand_list(sorted_intent))

                    # sorted_kb = list(Evaluator.expand_list(sorted_kb))
                    text_var = Variable(torch.LongTensor(padded_text))
                    slot_var = Variable(torch.LongTensor(list(Evaluator.expand_list(sorted_slot))))
                    intent_var = Variable(torch.LongTensor(sorted_intent))
                    kb_var = Variable(torch.LongTensor(sorted_kb))  #14*40*3
                    text_triple_var = Variable(torch.LongTensor(sorted_text_triple))
                    history_var = Variable(torch.LongTensor(sorted_history)) # 14*6*25   总共14句话,每句话的history中有六句话,每个history的话中有25个词


                    if torch.cuda.is_available():
                        text_var = text_var.cuda()
                        slot_var = slot_var.cuda()
                        intent_var = intent_var.cuda()
                        history_var = history_var.cuda()
                        kb_var =kb_var.cuda()

                    # slot_out, intent_out, prev_slot, prev_intent = self.__model(text_var, kb=kb_var, history=history_var, text_triple=text_triple_var, seq_lens=seq_lens,forced_slot=slot_var, forced_intent=intent_var, turn_index=turn_index,  if_train =True, prev_slot=prev_slot_tensor, prev_intent=prev_intent_tensor)

                    # 81%的epoch 有slot和intent的标注来训练，剩下9%+9%分别是只有slot和intent，1%啥都没有
                    random_slot, random_intent = random.random(), random.random()
                    # # 1 都有
                    if random_slot < self.__dataset.slot_forcing_rate and random_intent < self.__dataset.intent_forcing_rate:
                        slot_out, intent_out, prev_slot, prev_intent = self.__model(text_var, kb=kb_var, history=history_var, text_triple=text_triple_var, seq_lens=seq_lens,forced_slot=slot_var, forced_intent=intent_var, turn_index=turn_index,  if_train =True, prev_slot=prev_slot_tensor, prev_intent=prev_intent_tensor)
                    # 2 只有slot
                    elif random_slot < self.__dataset.slot_forcing_rate:
                        slot_out, intent_out, prev_slot, prev_intent = self.__model(text_var, kb=kb_var, history=history_var, text_triple=text_triple_var, seq_lens=seq_lens,forced_slot=slot_var , turn_index=turn_index,  if_train =True, prev_slot=prev_slot_tensor, prev_intent=prev_intent_tensor)
                    # 3 只有intent
                    elif random_intent < self.__dataset.intent_forcing_rate:
                        slot_out, intent_out, prev_slot, prev_intent = self.__model(text_var, kb=kb_var, history=history_var, text_triple=text_triple_var, seq_lens=seq_lens , forced_intent=intent_var, turn_index=turn_index,  if_train =True, prev_slot=prev_slot_tensor, prev_intent=prev_intent_tensor)
                    # 4 啥玩意都没有
                    else:
                        slot_out, intent_out, prev_slot, prev_intent = self.__model(text_var, kb=kb_var, history=history_var, text_triple=text_triple_var, seq_lens=seq_lens, turn_index=turn_index,  if_train =True, prev_slot=prev_slot_tensor, prev_intent=prev_intent_tensor)


                    "------------extrac local-------------------------"
                    prev_slot_list.append(prev_slot)
                    prev_intent_list.append(prev_intent)

                    turn_slot_loss = self.__criterion(slot_out, slot_var)  # slot_out.size=130*15     slot_var.size=130
                    turn_intent_loss = self.__criterion(intent_out, intent_var)

                    turn_loss = turn_slot_loss + turn_intent_loss
                    turn_loss = turn_loss/len(data_detail_batch) #loss regularization
                    turn_loss.backward()

                self.__optimizer.step()
                self.__optimizer.zero_grad()

                # try:
                #     total_slot_loss += slot_loss.cpu().item()
                #     total_intent_loss += intent_loss.cpu().item()
                # except AttributeError:
                #     total_slot_loss += slot_loss.cpu().data.numpy()[0]
                #     total_intent_loss += intent_loss.cpu().data.numpy()[0]


            time_con = time.time() - time_start
            # print('[Epoch {:2d}]: TRAIN: slot loss is {:2.6f}, intent loss is {:2.6f}, cost time about {:2.6} seconds.'.format(epoch, total_slot_loss, total_intent_loss, time_con))
            print('[Epoch {:2d}]: TRAIN: cost time about {:2.6} seconds.'.format(epoch, time_con))

            change, time_start = False, time.time()
            print("-------------------------DEV----------------------------------")
            dev_f1_score, dev_acc, dev_sent_acc = self.estimate(if_dev=True, test_batch=self.__batch_size)

            print("-------------------------TEST----------------------------------")
            # if dev_f1_score > best_dev_slot or dev_acc > best_dev_intent or dev_sent_acc > best_dev_sent:
            if dev_f1_score > best_dev_slot:
                test_f1, test_acc, test_sent_acc = self.estimate(if_dev=False, test_batch=self.__batch_size)

                if dev_f1_score > best_dev_slot: best_dev_slot = dev_f1_score
                if dev_acc > best_dev_intent: best_dev_intent = dev_acc
                if dev_sent_acc > best_dev_sent: best_dev_sent = dev_sent_acc

                print('\n[Epoch {:2d}]: TEST: slot f1: {:.3f}, intent acc: {:.3f}, semantic '
                      'acc: {:.3f}.'.format(epoch, test_f1, test_acc, test_sent_acc))
                # 保存model
                model_save_dir = os.path.join(self.__dataset.save_dir, "model")
                if not os.path.exists(model_save_dir):
                    os.mkdir(model_save_dir)
                torch.save(self.__model, os.path.join(model_save_dir, "model_"+str(train_index)+".pkl"))
                torch.save(self.__dataset, os.path.join(model_save_dir, "dataset_"+str(train_index)+".pkl"))

                time_con = time.time() - time_start
                # validation
                print('[Epoch {:2d}]: DEV : slot f1：{:2.6f}, ' \
                      'intent acc：{:2.6f}, semantic acc：{:.2f}, cost about ' \
                      '{:2.6f} seconds.'.format(epoch, dev_f1_score, dev_acc, dev_sent_acc, time_con))

            if train_index=='0':
                model_save_dir = os.path.join(self.__dataset.save_dir, "model")
                if not os.path.exists(model_save_dir):
                    os.mkdir(model_save_dir)
                torch.save(self.__model, os.path.join(model_save_dir, "model_0.pkl"))
                torch.save(self.__dataset, os.path.join(model_save_dir, "dataset_0.pkl"))

    def estimate(self, if_dev, test_batch=100):
        """
        Estimate the performance of model on dev or test dataset.
        """

        if if_dev:
            pred_slot, real_slot, pred_intent, real_intent, _ = self.prediction(
                self.__model, self.__dataset, "test", test_batch
            )
        else:
            pred_slot, real_slot, pred_intent, real_intent, _ = self.prediction(
                self.__model, self.__dataset, "dev", test_batch
            )
        # 用于计算F1


        slot_f1_socre = miulab.computeF1Score(pred_slot, real_slot)[0]
        # slot_f1_socre = Evaluator.f1_score(pred_slot, real_slot)

        intent_acc = Evaluator.accuracy(pred_intent, real_intent)
        sent_acc = Evaluator.semantic_acc(pred_slot, real_slot, pred_intent, real_intent)

        return slot_f1_socre, intent_acc, sent_acc

    @staticmethod
    def validate(model_path, dataset_path, batch_size):
        """
        validation will write mistaken samples to files and make scores.
        """

        model = torch.load(model_path)
        dataset = torch.load(dataset_path)

        # Get the sentence list in test dataset.
        sent_list = dataset.test_sentence

        pred_slot, real_slot, exp_pred_intent, real_intent, pred_intent = Processor.prediction(
            model, dataset, "dev", batch_size
        )

        # To make sure the directory for save error prediction.
        mistake_dir = os.path.join(dataset.save_dir, "error")
        if not os.path.exists(mistake_dir):
            os.mkdir(mistake_dir)

        slot_file_path = os.path.join(mistake_dir, "slot.txt")
        intent_file_path = os.path.join(mistake_dir, "intent.txt")
        both_file_path = os.path.join(mistake_dir, "both.txt")

        # 暂时封印这个, 后期改进的时候再弄
        # # Write those sample with mistaken slot prediction.
        # with open(slot_file_path, 'w') as fw:
        #     for w_list, r_slot_list, p_slot_list in zip(sent_list, real_slot, pred_slot):
        #         if r_slot_list != p_slot_list:
        #             for w, r, p in zip(w_list, r_slot_list, p_slot_list):
        #                 fw.write(w + '\t' + r + '\t' + p + '\n')
        #             fw.write('\n')
        #
        # # Write those sample with mistaken intent prediction.
        # with open(intent_file_path, 'w') as fw:
        #     for w_list, p_intent_list, r_intent, p_intent in zip(sent_list, pred_intent, real_intent, exp_pred_intent):
        #         if p_intent != r_intent:
        #             for w, p in zip(w_list, p_intent_list):
        #                 fw.write(w + '\t' + p + '\n')
        #             fw.write(r_intent + '\t' + p_intent + '\n\n')
        #
        # # Write those sample both have intent and slot errors.
        # with open(both_file_path, 'w') as fw:
        #     for w_list, r_slot_list, p_slot_list, p_intent_list, r_intent, p_intent in \
        #             zip(sent_list, real_slot, pred_slot, pred_intent, real_intent, exp_pred_intent):
        #
        #         if r_slot_list != p_slot_list or r_intent != p_intent:
        #             for w, r_slot, p_slot, p_intent_ in zip(w_list, r_slot_list, p_slot_list, p_intent_list):
        #                 fw.write(w + '\t' + r_slot + '\t' + p_slot + '\t' + p_intent_ + '\n')
        #             fw.write(r_intent + '\t' + p_intent + '\n\n')

        # print(pred_slot)
        # print(real_slot)

        slot_f1 = miulab.computeF1Score(pred_slot, real_slot)[0]
        # slot_f1 = Evaluator.f1_score(pred_slot, real_slot)
        intent_acc = Evaluator.accuracy(exp_pred_intent, real_intent)
        sent_acc = Evaluator.semantic_acc(pred_slot, real_slot, exp_pred_intent, real_intent)
        # sk_slot_f1 = f1_score(real_slot,pred_slot)

        return slot_f1, intent_acc, sent_acc

    @staticmethod
    def prediction(model, dataset, mode, batch_size):
        model.eval()

        if mode == "dev":
            dataloader = dataset.batch_delivery('dev', batch_size=batch_size, shuffle=False, is_digital=False)
        elif mode == "test":
            dataloader = dataset.batch_delivery('test', batch_size=batch_size, shuffle=False, is_digital=False)
        else:
            raise Exception("Argument error! mode belongs to {\"dev\", \"test\"}.")

        pred_slot, real_slot = [], []
        pred_intent, real_intent = [], []

        textPAD = ['<PAD>', '<PAD>', '<PAD>']

        max_column_len = 9
        slot_num = 3
        slot_dim, intent_dim = 128, 128

        for data_detail_batch in tqdm(dataloader, ncols=50):
            dial_id_batch, turn_id_batch, history_batch, slot_batch, text_batch, intent_batch, kb_batch, text_triple_batch, uuid_batch = [], [], [], [], [], [], [], [], []  # claim here

            prev_intent_list = [torch.zeros(1, intent_dim).cuda()]
            prev_slot_list = [torch.zeros(1, slot_dim).cuda()]

            # print("==================")
            digit_intent_list = []
            text_intent_list = []
            local_list = [[]for _ in range(batch_size)]
            for turn_index, turn in enumerate(data_detail_batch):

                # 以turn为基本单位, 但是每个turn中含有该batch中所有的dialogue对应的turn 以这个indent写
                dial_id_batch, turn_id_batch, history_batch, slot_batch, text_batch, intent_batch, kb_batch, text_triple_batch, uuid_batch = [], [], [], [], [], [], [], [], []  # init here

                for t in turn:
                    if len(t) != 0:
                        dial_id_batch.append(t['dial_id'])
                        turn_id_batch.append(t['turn_id'])
                        history_batch.append(t['history'])
                        slot_batch.append(t['slot'])
                        text_batch.append(t['text'])
                        intent_batch.append(t['intent'])
                        kb_batch.append(t['kb'])
                        text_triple_batch.append(t['triple'])
                        uuid_batch.append(t['uuid'])
                    else:
                        dial_id_batch.append(0)
                        turn_id_batch.append(0)
                        history_batch.append([['<PAD>']])
                        slot_batch.append(['<PAD>'])
                        text_batch.append(['<PAD>'])
                        intent_batch.append(['<PAD>'])
                        kb_batch.append([textPAD])
                        text_triple_batch.append([textPAD])
                        uuid_batch.append('0')

                # prev_slot_tensor = Variable(torch.cat(prev_slot_list, dim=0), requires_grad=True)
                # prev_intent_tensor = Variable(torch.cat(prev_intent_list, dim=0), requires_grad=True)
                prev_slot_tensor = Variable(prev_slot_list[-1], requires_grad=True)
                prev_intent_tensor = Variable(prev_intent_list[-1], requires_grad=True)

                padded_text, seq_lens, sorted_dial_id, sorted_turn_id, \
                sorted_history, sorted_slot, sorted_intent, sorted_kb, sorted_text_triple = dataset.add_padding(
                    text_batch, dial_id_batch, turn_id_batch,
                    history_batch, slot_batch, intent_batch, kb_batch, text_triple_batch, digital=False
                )
                real_slot.extend(sorted_slot)
                real_intent.extend(list(Evaluator.expand_list(sorted_intent)))
                digit_text = dataset.word_alphabet.get_index(padded_text)

                var_text = Variable(torch.LongTensor(digit_text))

                # ---------------add here-------------------
                digit_kb = dataset.word_alphabet.get_index(sorted_kb)
                var_kb = Variable(torch.LongTensor(digit_kb))

                # 按照kb的思路,将text_triple也加进去
                digit_text_triple = dataset.word_alphabet.get_index(sorted_text_triple)
                var_text_triple = Variable(torch.LongTensor(digit_text_triple))

                digit_history = dataset.word_alphabet.get_index(sorted_history)
                var_history = Variable(torch.LongTensor(digit_history))

                digit_local = dataset.word_alphabet.get_index(local_list)

                # if turn_index != 0:
                #     local_var = Variable(torch.LongTensor(digit_local))
                # else:
                #     local_var = Variable(torch.zeros(batch_size, max_column_len, slot_num))
                #     local_var = local_var.long()
                if torch.cuda.is_available():
                    var_text = var_text.cuda()
                    var_kb = var_kb.cuda()
                    var_history = var_history.cuda()
                    var_text_triple= var_text_triple.cuda()
                    # local_var =local_var.cuda()

                # 在这里重新输入到model中
                slot_idx, intent_idx, prev_slot, prev_intent = model(text=var_text, kb=var_kb, history= var_history, text_triple=var_text_triple, seq_lens=seq_lens, n_predicts=1, turn_index=turn_index, if_train=False , prev_slot=prev_slot_tensor, prev_intent=prev_intent_tensor )

                # add local knowledge for next turn
                prev_slot_list.append(prev_slot)
                prev_intent_list.append(prev_intent)

                nested_intent = Evaluator.nested_list([list(Evaluator.expand_list(intent_idx))], seq_lens)[0]
                pred_intent.extend(dataset.intent_alphabet.get_instance(nested_intent))

                # print(nested_intent)
                intent_index = Evaluator.max_freq_predict(nested_intent)
                nested_slot = Evaluator.nested_list([list(Evaluator.expand_list(slot_idx))], seq_lens)[0]
                # curr_slot_alphabet = dataset.slot_alphabet_list[intent_index[0]]
                pred_slot.extend(dataset.slot_alphabet.get_instance(nested_slot))

                # print(slot_batch)
                # print(slot_idx)
                # print(curr_slot_alphabet)

                "--------------extract local----------------------"
                # intent2column = {
                #     "navigate": ["poi", "poi_type", "address", "distance", "traffic_info"],
                #     "schedule": ["event", "time", "date", "room", "agenda", "party"],
                #     "weather": ["location", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "today"]
                # }
                #
                # slot_list = dataset.slot_alphabet.get_instance(nested_slot)
                # if turn_index==0:
                #     digit_intent_list = Evaluator.max_freq_predict(nested_intent)
                #     text_intent_list = dataset.intent_alphabet.get_instance(digit_intent_list)
                #
                # for sen_index,[sen_intent,sen_slot, sen_text] in enumerate(zip(digit_intent_list, slot_list, text_batch)):
                #     if turn_index == 0:
                #         curr_local = [[] for _ in range(max_column_len)]
                #     else:
                #         curr_local = local_list[sen_index]
                #
                #     curr_intent = dataset.intent_alphabet.get_instance(sen_intent)
                #     # print("{}  {}{}  {}  {}".format(sen_index, sen_intent, curr_intent, sen_slot, sen_text))
                #     # local[sent_index][sen_intent]
                #     column = intent2column[curr_intent]
                #     # print(column)
                #
                #     for word_index, [word_slot, word_text] in enumerate(zip(sen_slot, sen_text)):
                #         # print("{} {}".format(word_slot, word_text))
                #         slot_sep = word_slot.split('-')
                #         prefix, postfix = slot_sep[0], slot_sep[1] if len(slot_sep)!=1 else "O"
                #         if postfix in column and prefix=="B":
                #             slot_index = column.index(postfix)  #不需要知道具体index是什么含义，只需要能同一个含义对应同一个index就行
                #             if len(curr_local[slot_index])< slot_num:
                #                 curr_local[slot_index].append(word_text)    #暂定等于 其实可以append todo
                #             # print("{} {} {} {}".format(word_index, prefix, postfix, word_text))
                #     local_list[sen_index] = curr_local
                #
                #     # todo 每一轮的udpate的机制
                # # padding
                # for curr_local_index, curr_local in enumerate(local_list):
                #     for single_local_index, single_local in enumerate(curr_local):
                #         if len(single_local) < slot_num:
                #             padded_local = single_local
                #             padded_local = padded_local.extend([0] * (slot_num - len(single_local)))
                # # print("slot {} intent {}".format(pred_slot, pred_intent))
                # # print("-------------")
                #
                # # batch padding
                # for sen_local_index, sen_local in enumerate(local_list):
                #     batch_padding = [[0] * slot_num] * max_column_len
                #     if (len(sen_local) == 0):
                #         local_list[sen_local_index] = batch_padding

        exp_pred_intent = Evaluator.max_freq_predict(pred_intent)


        return pred_slot, real_slot, exp_pred_intent, real_intent, pred_intent


class Evaluator(object):

    @staticmethod
    def semantic_acc(pred_slot, real_slot, pred_intent, real_intent):
        """
        Compute the accuracy based on the whole predictions of
        given sentence, including slot and intent.
        """

        total_count, correct_count = 0.0, 0.0
        for p_slot, r_slot, p_intent, r_intent in zip(pred_slot, real_slot, pred_intent, real_intent):

            if p_slot == r_slot and p_intent == r_intent:
                correct_count += 1.0
            total_count += 1.0

        return 1.0 * correct_count / total_count

    @staticmethod
    def accuracy(pred_list, real_list):
        """
        Get accuracy measured by predictions and ground-trues.
        """

        pred_array = np.array(list(Evaluator.expand_list(pred_list)))
        real_array = np.array(list(Evaluator.expand_list(real_list)))
        return (pred_array == real_array).sum() * 1.0 / len(pred_array)

    @staticmethod
    def f1_score(pred_list, real_list):
        """
        Get F1 score measured by predictions and ground-trues.
        """

        tp, fp, fn = 0.0, 0.0, 0.0
        for i in range(len(pred_list)):
            seg = set()
            result = [elem.strip() for elem in pred_list[i]]
            target = [elem.strip() for elem in real_list[i]]

            j = 0
            while j < len(target):
                cur = target[j]
                if cur[0] == 'B':
                    k = j + 1
                    while k < len(target):
                        str_ = target[k]
                        if not (str_[0] == 'I' and cur[1:] == str_[1:]):
                            break
                        k = k + 1
                    seg.add((cur, j, k - 1))
                    j = k - 1
                j = j + 1

            tp_ = 0
            j = 0
            while j < len(result):
                cur = result[j]
                if cur[0] == 'B':
                    k = j + 1
                    while k < len(result):
                        str_ = result[k]
                        if not (str_[0] == 'I' and cur[1:] == str_[1:]):
                            break
                        k = k + 1
                    if (cur, j, k - 1) in seg:
                        tp_ += 1
                    else:
                        fp += 1
                    j = k - 1
                j = j + 1

            fn += len(seg) - tp_
            tp += tp_

        p = tp / (tp + fp) if tp + fp != 0 else 0
        r = tp / (tp + fn) if tp + fn != 0 else 0
        return 2 * p * r / (p + r) if p + r != 0 else 0

    """
    Max frequency prediction. 
    """

    @staticmethod
    def max_freq_predict(sample):
        predict = []
        for items in sample:
            predict.append(Counter(items).most_common(1)[0][0])
        return predict

    @staticmethod
    def exp_decay_predict(sample, decay_rate=0.8):
        predict = []
        for items in sample:
            item_dict = {}
            curr_weight = 1.0
            for item in items[::-1]:
                item_dict[item] = item_dict.get(item, 0) + curr_weight
                curr_weight *= decay_rate
            predict.append(sorted(item_dict.items(), key=lambda x_: x_[1])[-1][0])
        return predict

    @staticmethod
    def expand_list(nested_list):
        for item in nested_list:
            if isinstance(item, (list, tuple)):
                for sub_item in Evaluator.expand_list(item):
                    yield sub_item
            else:
                yield item

    @staticmethod
    def nested_list(items, seq_lens):
        num_items = len(items)
        trans_items = [[] for _ in range(0, num_items)]

        count = 0
        for jdx in range(0, len(seq_lens)):
            for idx in range(0, num_items):
                trans_items[idx].append(items[idx][count:count + seq_lens[jdx]])
            count += seq_lens[jdx]

        return trans_items
