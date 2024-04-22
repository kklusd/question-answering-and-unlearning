from model.DownstreamTasks.BertForQuestionAnswering import BertForQuestionAnswering
from flask import Flask, request, render_template
import json
import os
from model.BasicBert.BertConfig import BertConfig
from utils.data_helpers import LoadSQuADQuestionAnsweringDataset
from transformers import BertTokenizer
import torch
import collections
from tqdm import tqdm

app = Flask(__name__)
class ModelConfig:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_dir = os.path.join(self.project_dir, 'data', 'SQuAD')
        self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_uncased_english")
        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.train_file_path = os.path.join(self.dataset_dir, 'train-v1.1.json')
        self.retain_file_path = os.path.join(self.dataset_dir, 'retain-v1.1.json')
        self.forget_file_path = os.path.join(self.dataset_dir, 'forget-v1.1.json')
        self.test_file_path = os.path.join(self.dataset_dir, 'test-v1.1.json')
        self.val_file_path = os.path.join(self.dataset_dir, "dev-v1.1.json")
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.model_save_path = os.path.join(self.model_save_dir, 'model.pt')
        self.n_best_size = 10  # 对预测出的答案近后处理时，选取的候选答案数量
        self.max_answer_len = 30  # 在对候选进行筛选时，对答案最大长度的限制
        self.is_sample_shuffle = True  # 是否对训练集进行打乱
        self.use_torch_multi_head = False  # 是否使用PyTorch中的multihead实现
        self.batch_size = 12
        self.max_sen_len = 384  # 最大句子长度，即 [cls] + question ids + [sep] +  context ids + [sep] 的长度
        self.max_query_len = 64  # 表示问题的最大长度，超过长度截取
        self.learning_rate = 3.5e-5
        self.doc_stride = 128  # 滑动窗口一次滑动的长度
        self.epochs = 2
        self.model_val_per_epoch = 1
        bert_config_path = os.path.join(self.pretrained_model_dir, "config.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value
config = ModelConfig()
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
model = BertForQuestionAnswering(config, config.pretrained_model_dir)
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
        if inference:
            return all_results
        return acc_sum / (2 * n)

def write_prediction(test_iter, all_examples, logits_data):
    """
    根据预测得到的logits将预测结果写入到本地文件中
    :param test_iter:
    :param all_examples:
    :param logits_data:
    :return:
    """
    qid_to_example_context = {}  # 根据qid取到其对应的context token
    for example in all_examples:
        context = example[3]
        context_list = context.split()
        qid_to_example_context[example[0]] = context_list
    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["text", "start_index", "end_index", "start_logit", "end_logit"])
    prelim_predictions = collections.defaultdict(list)
    for b_input, _, _, b_qid, _, b_feature_id, b_map in tqdm(test_iter, ncols=80, desc="正在遍历候选答案"):
        # 取一个问题对应所有特征样本的预测logits（因为有了滑动窗口，所以原始一个context可以构造得到多个训练样子本）
        all_logits = logits_data[b_qid[0]]
        for logits in all_logits:
            if logits[0] != b_feature_id[0]:
                continue  # 非当前子样本对应的logits忽略
            # 遍历每个子样本对应logits的预测情况
            start_indexes = data_loader.get_best_indexes(logits[1], data_loader.n_best_size)
            # 得到开始位置几率最大的值对应的索引，例如可能是 [ 4,6,3,1]
            end_indexes = data_loader.get_best_indexes(logits[2], data_loader.n_best_size)
            # 得到结束位置几率最大的值对应的索引，例如可能是 [ 5,8,10,9]
            for start_index in start_indexes:
                for end_index in end_indexes:  # 遍历所有存在的结果组合
                    if start_index >= b_input.size(0):
                        continue  # 起始索引大于token长度，忽略
                    if end_index >= b_input.size(0):
                        continue  # 结束索引大于token长度，忽略
                    if start_index not in b_map[0]:
                        continue  # 用来判断索引是否位于[SEP]之后的位置，因为答案只会在[SEP]以后出现
                    if end_index not in b_map[0]:
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > data_loader.max_answer_length:
                        continue
                    token_ids = b_input.transpose(0, 1)[0]
                    strs = [data_loader.vocab.itos[s] for s in token_ids]
                    tok_text = " ".join(strs[start_index:(end_index + 1)])
                    tok_text = tok_text.replace(" ##", "").replace("##", "")
                    tok_text = tok_text.strip()
                    tok_text = " ".join(tok_text.split())

                    orig_doc_start = b_map[0][start_index]
                    orig_doc_end = b_map[0][end_index]
                    orig_tokens = qid_to_example_context[b_qid[0]][orig_doc_start:(orig_doc_end + 1)]
                    orig_text = " ".join(orig_tokens)
                    final_text = data_loader.get_final_text(tok_text, orig_text)

                    prelim_predictions[b_qid[0]].append(_PrelimPrediction(
                        text=final_text,
                        start_index=int(start_index),
                        end_index=int(end_index),
                        start_logit=float(logits[1][start_index]),
                        end_logit=float(logits[2][end_index])))
                    # 此处为将每个qid对应的所有预测结果放到一起，因为一个qid对应的context应该滑动窗口
                    # 会有构造得到多个训练样本，而每个训练样本都会对应得到一个预测的logits
                    # 并且这里取了n_best个logits，所以组合后一个问题就会得到过个预测的答案

    for k, v in prelim_predictions.items():
        # 对每个qid对应的所有预测答案按照start_logit+end_logit的大小进行排序
        prelim_predictions[k] = sorted(prelim_predictions[k],
                                        key=lambda x: (x.start_logit + x.end_logit),
                                        reverse=True)
    best_results = {}
    for k, v in prelim_predictions.items():
        best_results[k] = v[0].text  # 取最好的第一个结果
    return best_results


@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/inference", methods=["POST"])
def inference():
    formal_json = {
            "data": [
                {
                "title": "api test",
                "paragraphs": [
                    {
                    "context": "Super",
                    "qas": [
                        {
                        "answers": [
                            {
                            "answer_start": 1,
                            "text": "Super"
                            }
                        ],
                        "question": "",
                        "id": "56be4db0acb8001400a502ec"
                        }
                    ]
                    }
                ]
                }
            ],
            "version": "1.1"
            }
    if request.method == "POST":
        model_name = request.form.get("model")
        question = request.form.get("question")
        context = request.form.get("context")
        formal_json["data"][0]["paragraphs"][0]["context"] = context
        formal_json["data"][0]["paragraphs"][0]["qas"][0]["question"] = question
        filename = config.test_file_path
        with open(filename, "w") as f:
            json.dump(formal_json, f)
            f.close()
        test_iter, all_examples = data_loader.load_train_val_test_data(test_file_path=config.test_file_path,
                                                                       only_test=True)
        os.remove(filename)
        os.remove(os.path.join(config.dataset_dir, "cache_test-v11_doc_stride128_max_query_length64_max_sen_len384.pt"))
        model_save_path = os.path.join(config.model_save_dir, model_name+"_model.pt")
        print(model_save_path)
        loaded_paras = torch.load(model_save_path, map_location=config.device)
        model.load_state_dict(loaded_paras)
        model.to(config.device)
        all_result_logits = evaluate(test_iter, model, config.device,
                                 data_loader.PAD_IDX, inference=True)

        best_answer = write_prediction(test_iter, all_examples,
                                    all_result_logits)
        final_answer = ''
        for k, v in best_answer.items():
            final_answer = v
        return render_template("index.html", prediction_text="The answer is: {}".format(final_answer))
    
if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000)