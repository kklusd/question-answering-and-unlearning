import collections
import sys
sys.path.append('../')
from model.BasicBert.BertConfig import BertConfig
from model.DownstreamTasks.BertForQuestionAnswering import BertForQuestionAnswering
from utils.data_helpers import LoadSQuADQuestionAnsweringDataset
from utils.unlearning_loss import unlearn_loss
from utils.log_helper import logger_init
from transformers import BertTokenizer
from transformers import get_scheduler
import logging
import torch
import os
import time


class ModelConfig:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_dir = os.path.join(self.project_dir, 'data', 'SQuAD')
        self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_uncased_english")
        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.test_file_path = os.path.join(self.dataset_dir, 'dev-v1.1.json')
        self.forget_file_path = os.path.join(self.dataset_dir, 'forget-v1.1.json')
        self.retain_file_path = os.path.join(self.dataset_dir, 'retain-v1.1.json')
        self.val_file_path = os.path.join(self.dataset_dir, 'dev-v1.1.json')
        self.forget_ids_path = os.path.join(self.dataset_dir, 'forget_ids.json')
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.original_model_path = os.path.join(self.model_save_dir, 'original_model.pt')
        self.unlearn_model_path = os.path.join(self.model_save_dir, 'gradient_ascent_model.pt')
        self.n_best_size = 10  # 对预测出的答案近后处理时，选取的候选答案数量
        self.max_answer_len = 30  # 在对候选进行筛选时，对答案最大长度的限制
        self.is_sample_shuffle = True  # 是否对训练集进行打乱
        self.use_torch_multi_head = False  # 是否使用PyTorch中的multihead实现
        self.batch_size = 8
        self.max_sen_len = 384  # 最大句子长度，即 [cls] + question ids + [sep] +  context ids + [sep] 的长度
        self.max_query_len = 64  # 表示问题的最大长度，超过长度截取
        self.learning_rate = 1e-5
        self.doc_stride = 128  # 滑动窗口一次滑动的长度
        self.epochs = 1 # epoch of gradient ascent on forget dataset
        self.model_val_per_epoch = 1
        self.make_up_batches = 1 # number of selected retain samples to make up performance after unlearning
        self.make_up_retain = True
        logger_init(log_file_name='gradient_ascent', log_level=logging.DEBUG,
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

def main(config, only_for_eval=False):
    
    def train(config):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        lr_scheduler = get_scheduler(name='linear',
                            optimizer=optimizer,
                            num_warmup_steps=int(len(forget_iter) * 0),
                            num_training_steps=int(config.epochs * len(forget_iter)))
        max_acc = 0
        for epoch in range(config.epochs):
            start_time = time.time()
            forget_losses = 0
            losses = 0
            for idx, (batch_input, batch_seg, batch_label, _, _, _, _) in enumerate(forget_iter):
                loss = 0
                forget_batch_input = batch_input.to(config.device)  # [src_len, batch_size]
                forget_batch_seg = batch_seg.to(config.device)
                forget_batch_label = batch_label.to(config.device)
                forget_padding_mask = (forget_batch_input == data_loader.PAD_IDX).transpose(0, 1)
                forget_loss, forget_start_logits, forget_end_logits = model(input_ids=forget_batch_input,
                                                    attention_mask=forget_padding_mask,
                                                    token_type_ids=forget_batch_seg,
                                                    position_ids=None,
                                                    start_positions=forget_batch_label[:, 0],
                                                    end_positions=forget_batch_label[:, 1])
                loss -= forget_loss
                absolute_forget_loss = forget_loss.item()
                if config.make_up_retain:
                    assert config.make_up_batches > 0, "Make up batches must larger than 0!!!"
                    for idy, (batch_input, batch_seg, batch_label, _, _, _, _) in enumerate(retain_iter):
                        retain_batch_input = batch_input.to(config.device)  # [src_len, batch_size]
                        retain_batch_seg = batch_seg.to(config.device)
                        retain_batch_label = batch_label.to(config.device)
                        retain_padding_mask = (retain_batch_input == data_loader.PAD_IDX).transpose(0, 1)
                        retain_loss, retain_start_logits, retain_end_logits = model(input_ids=retain_batch_input,
                                                            attention_mask=retain_padding_mask,
                                                            token_type_ids=retain_batch_seg,
                                                            position_ids=None,
                                                            start_positions=retain_batch_label[:, 0],
                                                            end_positions=retain_batch_label[:, 1])
                        loss += retain_loss
                        if idy > config.make_up_batches:
                            break
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                losses += loss
                forget_losses += absolute_forget_loss
                if idx % 10 == 0:
                    logging.info(f"Epoch: {epoch}, Batch[{idx}/{len(forget_iter)}], "
                                f"Forget loss :{absolute_forget_loss:.3f}, "
                                f"Unlearn loss :{losses: .3f}")
                if idx % 100 == 0:
                    y_pred = [forget_start_logits.argmax(1), forget_end_logits.argmax(1)]
                    y_true = [forget_batch_label[:, 0], forget_batch_label[:, 1]]
                    show_result(forget_batch_input, data_loader.vocab.itos,
                                y_pred=y_pred, y_true=y_true)
            forget_loss = forget_losses / len(forget_iter)
            end_time = time.time()
            logging.info(f"Epoch: {epoch}, Forget loss: {forget_loss:.3f}, "
                         f"Epoch time = {(end_time - start_time):.3f}")
            if (epoch + 1) % config.model_val_per_epoch == 0:
                val_acc = evaluate(val_iter, model,
                            config.device,
                            data_loader.PAD_IDX,
                            inference=False)
                forget_acc = evaluate(forget_iter, model,
                            config.device,
                            data_loader.PAD_IDX,
                            inference=False)
                if val_acc > max_acc:
                    max_acc = val_acc
                logging.info(f" ### Accuracy on val: {round(val_acc, 4)} max :{max_acc}")
                logging.info(f" ### Accuracy on forget: {round(forget_acc, 4)}")
                torch.save(model.state_dict(), config.unlearn_model_path)
    bert_tokenize = BertTokenizer.from_pretrained(config.pretrained_model_dir).tokenize
    data_loader = LoadSQuADQuestionAnsweringDataset(vocab_path=config.vocab_path,
                                                    tokenizer=bert_tokenize,
                                                    batch_size=config.batch_size,
                                                    max_sen_len=config.max_sen_len,
                                                    max_query_length=config.max_query_len,
                                                    max_position_embeddings=config.max_position_embeddings,
                                                    pad_index=config.pad_token_id,
                                                    is_sample_shuffle=config.is_sample_shuffle,
                                                    doc_stride=config.doc_stride
                                                    )
    retain_iter, test_iter, val_iter, forget_iter = \
        data_loader.load_train_val_test_data(train_file_path=config.retain_file_path,
                                            test_file_path=config.test_file_path,
                                            forget_file_path=config.forget_file_path,
                                            val_file_path=config.val_file_path,
                                            unlearn=True,
                                            only_test=False)
    model = BertForQuestionAnswering(config, config.pretrained_model_dir)
    model = model.to(config.device)
    if only_for_eval:
        if os.path.exists(config.unlearn_model_path):
            loaded_paras = torch.load(config.unlearn_model_path)
            model.load_state_dict(loaded_paras)
            logging.info("## 成功载入已有模型，进行追加训练......")
        else:
            raise ValueError("You should have an unlearned model!!!")
        val_acc = evaluate(val_iter, model,
                            config.device,
                            data_loader.PAD_IDX,
                            inference=False)
        forget_acc = evaluate(forget_iter, model,
                    config.device,
                    data_loader.PAD_IDX,
                    inference=False)
        # retain_acc = evaluate(retain_iter, model, 
        #                       config.device, 
        #                       data_loader.PAD_IDX,
        #                       inference=False)
        print(f" ### Accuracy on val: {round(val_acc, 4)}")
        print(f" ### Accuracy on forget: {round(forget_acc, 4)}")
        # print(f" ### Accuracy on retain: {round(retain_acc, 4)}")
    else:
        if os.path.exists(config.original_model_path):
            loaded_paras = torch.load(config.original_model_path, map_location=config.device)
            model.load_state_dict(loaded_paras)
            logging.info("## 成功载入已有模型，进行追加训练......")
        else:
            raise ValueError("You should have a finetuned model!!!")
        train(config)
    


def evaluate(data_iter, model, device, PAD_IDX, inference=False):
    model.eval()
    with torch.no_grad():
        acc_sum, n = 0.0, 0
        all_results = collections.defaultdict(list)
        for batch_input, batch_seg, batch_label, batch_qid, _, batch_feature_id, _ in data_iter:
            batch_input = batch_input.to(device)  # [src_len, batch_size]
            batch_seg = batch_seg.to(device)
            batch_label = batch_label.to(device)
            padding_mask = (batch_input == PAD_IDX).transpose(0, 1)
            start_logits, end_logits = model(input_ids=batch_input,
                                             attention_mask=padding_mask,
                                             token_type_ids=batch_seg,
                                             position_ids=None)
            # 将同一个问题下的所有预测样本的结果保存到一个list中，这里只对batchsize=1时有用
            all_results[batch_qid[0]].append([batch_feature_id[0],
                                              start_logits.cpu().numpy().reshape(-1),
                                              end_logits.cpu().numpy().reshape(-1)])
            if not inference:
                acc_sum_start = (start_logits.argmax(1) == batch_label[:, 0]).float().sum().item()
                acc_sum_end = (end_logits.argmax(1) == batch_label[:, 1]).float().sum().item()
                acc_sum += (acc_sum_start + acc_sum_end)
                n += len(batch_label)
        model.train()
        if inference:
            return all_results
        return acc_sum / (2 * n)


def show_result(batch_input, itos, num_show=5, y_pred=None, y_true=None):
    """
    本函数的作用是在训练模型的过程中展示相应的结果
    :param batch_input:
    :param itos:
    :param num_show:
    :param y_pred:
    :param y_true:
    :return:
    """
    count = 0
    batch_input = batch_input.transpose(0, 1)  # 转换为[batch_size, seq_len]形状
    for i in range(len(batch_input)):  # 取一个batch所有的原始文本
        if count == num_show:
            break
        input_tokens = [itos[s] for s in batch_input[i]]  # 将question+context 的ids序列转为字符串
        start_pos, end_pos = y_pred[0][i], y_pred[1][i]
        answer_text = " ".join(input_tokens[start_pos:(end_pos + 1)]).replace(" ##", "")
        input_text = " ".join(input_tokens).replace(" ##", "").split('[SEP]')
        question_text, context_text = input_text[0], input_text[1]

        logging.info(f"### Question: {question_text}")
        logging.info(f"  ## Predicted answer: {answer_text}")
        start_pos, end_pos = y_true[0][i], y_true[1][i]
        true_answer_text = " ".join(input_tokens[start_pos:(end_pos + 1)])
        true_answer_text = true_answer_text.replace(" ##", "")
        logging.info(f"  ## True answer: {true_answer_text}")
        logging.info(f"  ## True answer idx: {start_pos.cpu(), end_pos.cpu()}")
        count += 1


def inference(config):
    bert_tokenize = BertTokenizer.from_pretrained(config.pretrained_model_dir).tokenize
    data_loader = LoadSQuADQuestionAnsweringDataset(vocab_path=config.vocab_path,
                                                    tokenizer=bert_tokenize,
                                                    batch_size=1,  # 只能是1
                                                    max_sen_len=config.max_sen_len,
                                                    doc_stride=config.doc_stride,
                                                    max_query_length=config.max_query_len,
                                                    max_answer_length=config.max_answer_len,
                                                    max_position_embeddings=config.max_position_embeddings,
                                                    pad_index=config.pad_token_id,
                                                    n_best_size=config.n_best_size)
    test_iter, all_examples = data_loader.load_train_val_test_data(test_file_path=config.test_file_path,
                                                                   only_test=True)
    model = BertForQuestionAnswering(config,
                                     config.pretrained_model_dir)
    if os.path.exists(config.unlearn_model_save_path):
        loaded_paras = torch.load(config.unlearn_model_save_path)
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，开始进行推理......")
    else:
        raise ValueError(f"## 模型{config.unlearn_model_save_path}不存在，请检查路径或者先训练模型......")

    model = model.to(config.device)
    all_result_logits = evaluate(test_iter, model, config.device,
                                 data_loader.PAD_IDX, inference=True)
    data_loader.write_prediction(test_iter, all_examples,
                                 all_result_logits, config.dataset_dir)


if __name__ == '__main__':
    model_config = ModelConfig()
    main(config=model_config, only_for_eval=False)
    # inference(model_config)
