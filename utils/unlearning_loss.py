import torch.nn as nn
import torch
import torch.nn.functional as F

def unlearn_loss(start_logits,
                 end_logits,
               start_positions=None,
                 end_positions=None,
                 forget_labels=None):
          b, c = start_logits.shape
          # 由于部分情况下start/end 位置会超过输入的长度
          # （例如输入序列的可能大于512，并且正确的开始或者结束符就在512之后）
          # 那么此时就要进行特殊处理
          ignored_index = start_logits.size(1)  # 取输入序列的长度
          start_positions.clamp_(0, ignored_index)
          # 如果正确起始位置start_positions中，存在输入样本的开始位置大于输入长度，
          # 那么直接取输入序列的长度作为开始位置
          end_positions.clamp_(0, ignored_index)
          start_logits = start_logits.view(b, c, -1)
          end_logits = end_logits.view(b, c, -1)
          start_positions = start_positions.view(b, -1)
          end_positions = end_positions.view(b, -1)
          loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index, reduction="none")
          # 这里指定ignored_index其实就是为了忽略掉超过输入序列长度的（起始结束）位置
          # 在预测时所带来的损失，因为这些位置并不能算是模型预测错误的（只能看做是没有预测），
          # 同时如果不加ignore_index的话，那么可能会影响模型在正常情况下的语义理解能力
          start_loss = loss_fct(start_logits, start_positions)
          end_loss = loss_fct(end_logits, end_positions)
          start_loss = start_loss.mean(-1)
          end_loss = end_loss.mean(-1)
          forget_size = torch.count_nonzero(forget_labels)
          if forget_size.item() == 0:
               forget_loss = 0
          else:
               forget_loss = torch.mean(forget_labels * start_loss + forget_labels * end_loss, dim=0) * (b / forget_size)
               forget_loss = 0.5 * forget_loss.detach().item()
          if (b - forget_size).item() == 0:
               retain_loss = 0
          else:
               retain_loss = torch.mean((1-forget_labels) * start_loss + (1-forget_labels) * end_loss, dim=0) * b / (b-forget_size)
               retain_loss = 0.5 * retain_loss.detach().item()
          start_loss = (1 - forget_labels) * start_loss - forget_labels * start_loss
          start_loss = torch.mean(start_loss, dim=0)
          end_loss = (1-forget_labels) * end_loss - forget_labels * end_loss
          end_loss = torch.mean(end_loss, dim=0)
          return (start_loss + end_loss) / 2, forget_loss, retain_loss

def distill_loss(student_start_logits, student_end_logits, teacher_start_logits, teacher_end_logits, forget_labels, KL_temperature=1):
     student_start_out = F.log_softmax(student_start_logits / KL_temperature, dim=1)
     student_end_out = F.log_softmax(student_end_logits / KL_temperature, dim=1)
     teacher_start_out = F.softmax(teacher_start_logits / KL_temperature, dim=1)
     teacher_end_out = F.softmax(teacher_end_logits / KL_temperature, dim=1)
     start_kl_loss = torch.sum(F.kl_div(student_start_out, teacher_start_out, reduction="none"), dim=1)
     end_kl_loss = torch.sum(F.kl_div(student_end_out, teacher_end_out, reduction="none"), dim=1)
     start_kl_loss = torch.mean((1-forget_labels) * start_kl_loss - forget_labels * start_kl_loss)
     end_kl_loss = torch.mean((1-forget_labels) * end_kl_loss - forget_labels * end_kl_loss)
     kl_loss = 0.5 * (start_kl_loss + end_kl_loss)

     return kl_loss