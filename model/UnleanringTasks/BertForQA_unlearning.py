from ..BasicBert.Bert import BertModel
import torch.nn as nn


class BertForQuestionAnswering(nn.Module):
    """
    用于建模类似SQuAD这样的问答数据集
    """

    def __init__(self, config, bert_pretrained_model_dir=None):
        super(BertForQuestionAnswering, self).__init__()
        if bert_pretrained_model_dir is not None:
            self.bert = BertModel.from_pretrained(config, bert_pretrained_model_dir)
        else:
            self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None):
        """
        :param input_ids: [src_len,batch_size]
        :param attention_mask: [batch_size,src_len]
        :param token_type_ids: [src_len,batch_size]
        :param position_ids:
        :return:
        """
        _, all_encoder_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)
        sequence_output = all_encoder_outputs[-1]  # 取Bert最后一层的输出
        # sequence_output: [src_len, batch_size, hidden_size]
        logits = self.qa_outputs(sequence_output)  # [src_len, batch_size,2]
        start_logits, end_logits = logits.split(1, dim=-1)
        # [src_len,batch_size,1]  [src_len,batch_size,1]
        start_logits = start_logits.squeeze(-1).transpose(0, 1)  # [batch_size,src_len]
        end_logits = end_logits.squeeze(-1).transpose(0, 1)  # [batch_size,src_len]
        return start_logits, end_logits  # [batch_size,src_len]
