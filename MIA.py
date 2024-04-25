import collections
import sys
from sklearn.svm import SVC
from tqdm import tqdm
sys.path.append('../')
from model.BasicBert.BertConfig import BertConfig
from model.UnleanringTasks.BertForQA_unlearning import BertForQuestionAnswering
from utils.data_helpers import LoadSQuADQuestionAnsweringDataset
from utils.unlearning_loss import unlearn_loss
from utils.log_helper import logger_init
from transformers import BertTokenizer
from transformers import get_scheduler
import logging
import torch
import os
import time
import numpy as np

class ModelConfig:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_dir = os.path.join(self.project_dir, 'data', 'SQuAD')
        self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_uncased_english")
        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.train_file_path = os.path.join(self.dataset_dir, 'train-v1.1.json')
        self.retain_file_path = os.path.join(self.dataset_dir, 'retain-v1.1.json')
        self.test_file_path = os.path.join(self.dataset_dir, 'dev-v1.1.json')
        self.val_file_path = os.path.join(self.dataset_dir, 'dev-v1.1.json')
        self.forget_file_path = os.path.join(self.dataset_dir, 'forget-v1.1.json')
        self.forget_ids_path = os.path.join(self.dataset_dir, 'forget_ids.json')
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.original_model_path = os.path.join(self.model_save_dir, 'original_model.pt')
        self.unlearn_model_path = os.path.join(self.model_save_dir, 'retrain_model.pt')
        self.n_best_size = 10  # 对预测出的答案近后处理时，选取的候选答案数量
        self.max_answer_len = 30  # 在对候选进行筛选时，对答案最大长度的限制
        self.is_sample_shuffle = True  # 是否对训练集进行打乱
        self.use_torch_multi_head = False  # 是否使用PyTorch中的multihead实现
        self.batch_size = 12
        self.max_sen_len = 384  # 最大句子长度，即 [cls] + question ids + [sep] +  context ids + [sep] 的长度
        self.max_query_len = 64  # 表示问题的最大长度，超过长度截取
        self.learning_rate = 3.5e-5
        self.doc_stride = 128  # 滑动窗口一次滑动的长度
        self.epochs = 1
        self.model_val_per_epoch = 1
        logger_init(log_file_name='gradient_difference', log_level=logging.DEBUG,
                    log_dir=self.logs_save_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        # 把原始bert中的配置参数也导入进来
        bert_config_path = os.path.join(self.pretrained_model_dir, "config.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value
        # 将当前配置打印到日志文件中
        logging.info(" ### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"### {key} = {value}")


def SVC_fit_predict(shadow_train, shadow_test, target_test):
    n_shadow_train = shadow_train.shape[0]
    n_shadow_test = shadow_test.shape[0]
    n_target_test = target_test.shape[0]

    X_shadow = np.concatenate((shadow_train,shadow_test))
    Y_shadow = np.concatenate([np.zeros(n_shadow_train), np.ones(n_shadow_test)])

    clf = SVC(C=3, gamma="auto", kernel="rbf")
    clf.fit(X_shadow, Y_shadow)

    accs = []

    if n_target_test > 0:
        acc_test = clf.predict(target_test).mean()
        accs.append(acc_test)

    return np.mean(accs)


def SVC_MIA(fear,feaf,feat):


    acc_raw = SVC_fit_predict(fear,feaf,feat)

    m = {
        'raw': acc_raw
    }
    return m





if __name__ == '__main__':
    config = ModelConfig()
    bert_tokenize = BertTokenizer.from_pretrained(config.pretrained_model_dir).tokenize
    data_loader = LoadSQuADQuestionAnsweringDataset(vocab_path=config.vocab_path,
                                                    tokenizer=bert_tokenize,
                                                    batch_size=config.batch_size,
                                                    max_sen_len=config.max_sen_len,
                                                    max_query_length=config.max_query_len,
                                                    max_position_embeddings=config.max_position_embeddings,
                                                    pad_index=config.pad_token_id,
                                                    is_sample_shuffle=config.is_sample_shuffle,
                                                    doc_stride=config.doc_stride,
                                                    forget_ids_path=config.forget_ids_path,
                                                    unlearning=True
                                                    )
    model = BertForQuestionAnswering(config, config.pretrained_model_dir)
    model = model.to(config.device)
    if os.path.exists(config.unlearn_model_path):
        loaded_paras = torch.load(config.unlearn_model_path, map_location=config.device)
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，进行追加训练......")
    else:
        raise ValueError("You should have an unlearned model!!!")
    model = model.to(config.device)
    retain_iter, test_iter, val_iter, forget_iter = \
        data_loader.load_train_val_test_data(train_file_path=config.retain_file_path,
                                             test_file_path=config.test_file_path,
                                             forget_file_path=config.forget_file_path,
                                             unlearn=True,
                                             only_test=False)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "")
    model.eval()
    PAD_IDX = data_loader.PAD_IDX
    with torch.no_grad():
        acc_sum, n = 0.0, 0
        all_results = collections.defaultdict(list)
        fear = np.zeros([1,768])
        feaf = np.zeros([1,768])
        feat = np.zeros([1,768])

        for batch_input, batch_seg, batch_label, batch_qid, _, batch_feature_id, _, _ in tqdm(retain_iter):
            batch_input = batch_input.to(device)  # [src_len, batch_size]
            batch_seg = batch_seg.to(device)
            batch_label = batch_label.to(device)
            padding_mask = (batch_input == PAD_IDX).transpose(0, 1)
            start_logits, end_logits = model(input_ids=batch_input,
                                             attention_mask=padding_mask,
                                             token_type_ids=batch_seg,
                                             position_ids=None)
            logita = start_logits.detach().cpu().numpy()
            logitb = end_logits.detach().cpu().numpy()
            new_fear = np.concatenate((logita, logitb),1)
            fear = np.concatenate((fear,new_fear))
            if fear.shape[0] > 2000:
                break

        for batch_input, batch_seg, batch_label, batch_qid, _, batch_feature_id, _, _ in tqdm(forget_iter):
            batch_input = batch_input.to(device)  # [src_len, batch_size]
            batch_seg = batch_seg.to(device)
            batch_label = batch_label.to(device)
            padding_mask = (batch_input == PAD_IDX).transpose(0, 1)
            start_logits, end_logits = model(input_ids=batch_input,
                                             attention_mask=padding_mask,
                                             token_type_ids=batch_seg,
                                             position_ids=None)
            logita = start_logits.detach().cpu().numpy()
            logitb = end_logits.detach().cpu().numpy()
            new_fea = np.concatenate((logita, logitb), 1)
            feaf = np.concatenate((feaf, new_fea))
            if feaf.shape[0] > 2000:
                break
        for batch_input, batch_seg, batch_label, batch_qid, _, batch_feature_id, _, _ in tqdm(test_iter):
            batch_input = batch_input.to(device)  # [src_len, batch_size]
            batch_seg = batch_seg.to(device)
            batch_label = batch_label.to(device)
            padding_mask = (batch_input == PAD_IDX).transpose(0, 1)
            start_logits, end_logits = model(input_ids=batch_input,
                                             attention_mask=padding_mask,
                                             token_type_ids=batch_seg,
                                             position_ids=None)
            logita = start_logits.detach().cpu().numpy()
            logitb = end_logits.detach().cpu().numpy()
            new_fea = np.concatenate((logita, logitb), 1)
            feat = np.concatenate((feat, new_fea))
            if feat.shape[0] > 2000:
                break

    m = SVC_MIA(fear[1:,:], feaf[1:,:], feat[1:,:])
    print(m)
