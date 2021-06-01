import torch
from torch import nn
from torch.nn import Module
from torch.nn import functional as F
from torchcrf import CRF
from transformers import BertModel, BertPreTrainedModel
# from transformers import add_start_docstrings, add_start_docstrings_to_callable

from loss import FocalLoss, DSCLoss, DiceLoss, LabelSmoothingCrossEntropy
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss, BCEWithLogitsLoss

class BertCRFForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        # for [cls] and [sep], attention mask对应位置 改为 1
        
        if labels is not None:
            loss = -1. * self.crf(emissions=logits, tags=labels.long(), \
                mask=attention_mask.byte(), reduction='mean')
            preds = self.crf.decode(emissions=logits, mask=attention_mask.byte())
            outputs = (loss,) + (logits,) +(preds,) + outputs[2:]  # add hidden states and attention if they are here
            # 判断 seq_length 是否一致
            assert [len(pred) for pred in preds]==torch.sum(attention_mask, axis=1).tolist()
        return outputs  # (loss), scores, (hidden_states), (attentions)


class BertForSequenceMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        
        # label_count = torch.tensor([191, 104, 26, 83, 32, 128, 51, 64, 225, 1242, 151, 299, 197, 300,\
        #     154, 96, 63, 99, 69, 462, 325, 138, 2004, 145, 100, 109, 33, 63, 121, 65, 61, 298, 274, 134,\
        #     79, 107, 827, 238, 727, 99, 105, 82, 177, 170, 268, 75, 287, 128, 48, 210, 80, 122, 110, 145,\
        #     605, 356, 93, 82, 47, 87, 197, 67, 63, 254, 74], dtype=torch.float)
        # label_weight = label_count.sum()/label_count
        # label_weight += min(label_weight)*3
        # label_weight /= min(label_weight).clone()
        # label_weight= label_weight.cuda()

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            # loss_fct = BCEWithLogitsLoss(weight=label_weight)
            loss = loss_fct(logits, labels.float())
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertForTokenClassificationWithDiceLoss(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    # @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            
            # loss_fct = CrossEntropyLoss()
            # loss_fct = FocalLoss()
            loss_fct = DiceLoss()
            # loss_fct = DSCLoss()
            # loss_fct= LabelSmoothingCrossEntropy()

            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                # print(active_loss, active_loss.shape, \
                #      active_logits,active_logits.shape,\
                #      active_labels,active_labels.shape,\
                #      labels, labels.shape)
                #2048 2048*435 2048 8*256 
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)



from utils_ner_bin import get_entities
class BertForTokenBinaryClassificationJoint(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.trigger_num_labels = config.trigger_num_labels
        self.role_num_labels = config.role_num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.trigger_start_classifier = nn.Linear(config.hidden_size, config.trigger_num_labels)
        self.trigger_end_classifier = nn.Linear(config.hidden_size, config.trigger_num_labels)
        self.role_start_classifier = nn.Linear(config.hidden_size, config.role_num_labels)
        self.role_end_classifier = nn.Linear(config.hidden_size, config.role_num_labels)

        self.init_weights()

    # @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        trigger_start_labels=None, # batch * trigger_num_class * seq_length 
        trigger_end_labels=None,
        role_start_labels=None, # batch* role_num_class * seq_length
        role_end_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=torch.zeros_like(token_type_ids),
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        outputs= outputs[2:]

        #######################################################
        ## trigger
        sequence_output = self.dropout(sequence_output)
        trigger_start_logits = self.trigger_start_classifier(sequence_output)
        trigger_end_logits = self.trigger_end_classifier(sequence_output)

        if trigger_start_labels is not None and trigger_end_labels is not None:
            # loss_fct = CrossEntropyLoss()
            # loss_fct = FocalLoss(class_num=self.trigger_num_labels)
            loss_fct = BCEWithLogitsLoss(reduction="none")
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_trigger_start_logits = trigger_start_logits.view(-1, self.trigger_num_labels)
                active_trigger_end_logits = trigger_end_logits.view(-1, self.trigger_num_labels)

                active_trigger_start_labels = trigger_start_labels.view(-1, self.trigger_num_labels)
                active_trigger_end_labels = trigger_end_labels.view(-1, self.trigger_num_labels)

                trigger_start_loss = loss_fct(active_trigger_start_logits, active_trigger_start_labels.float())
                trigger_start_loss = trigger_start_loss * (active_loss.unsqueeze(-1))
                trigger_start_loss = torch.sum(trigger_start_loss)/torch.sum(active_loss)

                trigger_end_loss = loss_fct(active_trigger_end_logits, active_trigger_end_labels.float())
                trigger_end_loss = trigger_end_loss * (active_loss.unsqueeze(-1))
                trigger_end_loss = torch.sum(trigger_end_loss)/torch.sum(active_loss)

            else:
                trigger_start_loss = loss_fct(trigger_start_logits.view(-1, self.trigger_num_labels), trigger_start_labels.view(-1))
                trigger_end_loss = loss_fct(trigger_end_logits.view(-1, self.trigger_num_labels), trigger_end_labels.view(-1))
            trigger_loss = trigger_start_loss+ trigger_end_loss

        #######################################################
        ## role
        # add trigger embedding
        batch_size, sequence_length, hidden_size = sequence_output.shape
        mask = token_type_ids.unsqueeze(-1).expand_as(sequence_output).bool()
        trigger_embedding = torch.sum(sequence_output * mask, dim=1) / torch.sum(mask, dim=1)
        context_embedding = sequence_output + trigger_embedding.unsqueeze(1)
        
        # sequence_output_role = self.dropout(sequence_output)
        role_start_logits = self.role_start_classifier(context_embedding)
        role_end_logits = self.role_end_classifier(context_embedding)

        if role_start_labels is not None and role_end_labels is not None:
            # loss_fct = CrossEntropyLoss()
            # loss_fct = FocalLoss(class_num=self.role_num_labels)
            loss_fct = BCEWithLogitsLoss(reduction="none")
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_role_start_logits = role_start_logits.view(-1, self.role_num_labels)
                active_role_end_logits = role_end_logits.view(-1, self.role_num_labels)

                active_role_start_labels = role_start_labels.view(-1, self.role_num_labels)
                active_role_end_labels = role_end_labels.view(-1, self.role_num_labels)

                role_start_loss = loss_fct(active_role_start_logits, active_role_start_labels.float())
                role_start_loss = role_start_loss * (active_loss.unsqueeze(-1))
                role_start_loss = torch.sum(role_start_loss)/torch.sum(active_loss)

                role_end_loss = loss_fct(active_role_end_logits, active_role_end_labels.float())
                role_end_loss = role_end_loss * (active_loss.unsqueeze(-1))
                role_end_loss = torch.sum(role_end_loss)/torch.sum(active_loss)

            else:
                role_start_loss = loss_fct(role_start_logits.view(-1, self.role_num_labels), role_start_labels.view(-1))
                role_end_loss = loss_fct(role_end_logits.view(-1, self.role_num_labels), role_end_labels.view(-1))
            role_loss = role_start_loss+ role_end_loss
            
        outputs = (trigger_loss+ role_loss,) 

        return outputs  # (loss), scores, (hidden_states), (attentions)

    def add_trigger_embedding(self, trigger, context_embedding):
        # add trigger embedding
        _, start, end, label = trigger
        trigger_embedding = []
        for j in range(start, end+1):
            trigger_embedding.append(context_embedding[j])
        trigger_embedding = torch.stack(trigger_embedding,dim=0)
        trigger_embedding = torch.mean(trigger_embedding,dim=0)
        context_embedding += trigger_embedding
        return context_embedding
        
    def predict_trigger(self, sequence_output):
        #######################################################
        ## trigger
        # sequence_output_trigger = self.dropout(sequence_output)
        trigger_start_logits = self.trigger_start_classifier(sequence_output)
        trigger_end_logits = self.trigger_end_classifier(sequence_output)

        threshold = 0.5
        trigger_start_logits = torch.sigmoid(trigger_start_logits)> threshold # 1498*256*217
        trigger_end_logits = torch.sigmoid(trigger_end_logits) > threshold
        # 64*256*65
        trigger_list = get_entities(trigger_start_logits.cpu().numpy(), trigger_end_logits.cpu().numpy())[0]
        return trigger_list

    def predict_role(self, trigger, sequence_output):     
        context_embedding = self.add_trigger_embedding(trigger, sequence_output[0]).unsqueeze(0)

        role_start_logits = self.role_start_classifier(context_embedding)
        role_end_logits = self.role_end_classifier(context_embedding)
        
        threshold = 0.5
        role_start_logits = torch.sigmoid(role_start_logits)> threshold # 1498*256*217
        role_end_logits = torch.sigmoid(role_end_logits) > threshold

        role_list = get_entities(role_start_logits.cpu().numpy(), role_end_logits.cpu().numpy())[0]
        return role_list

    def predict(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        outputs = outputs[2:]

        result = []
        trigger_list = self.predict_trigger(sequence_output)[:10]
        for trigger in trigger_list:
            role_list = self.predict_role(trigger, sequence_output)
            role_list = [trigger] + role_list[:10]
            result.append(role_list)
        return result  # (loss), scores, (hidden_states), (attentions)


class BertForTokenBinaryClassificationMultiTask(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.trigger_num_labels = config.trigger_num_labels
        self.role_num_labels = config.role_num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.trigger_start_classifier = nn.Linear(config.hidden_size, config.trigger_num_labels)
        self.trigger_end_classifier = nn.Linear(config.hidden_size, config.trigger_num_labels)
        self.role_start_classifier = nn.Linear(config.hidden_size, config.role_num_labels)
        self.role_end_classifier = nn.Linear(config.hidden_size, config.role_num_labels)

        if self.config.with_gate:  
            self.global_classifier = nn.Linear(config.hidden_size, 1) # batch_size, max_seq_length, hidden_size -> batch_size, max_seq_length, 1
            self.activation = nn.ReLU()

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        trigger_start_labels=None, # batch * trigger_num_class * seq_length 
        trigger_end_labels=None,
        role_start_labels=None, # batch* role_num_class * seq_length
        role_end_labels=None,
    ):
       
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=torch.zeros_like(token_type_ids),
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        outputs= outputs[2:]

        #######################################################
        ## trigger
        sequence_output = self.dropout(sequence_output)
        trigger_start_logits = self.trigger_start_classifier(sequence_output)
        trigger_end_logits = self.trigger_end_classifier(sequence_output)

        if trigger_start_labels is not None and trigger_end_labels is not None:
            # loss_fct = CrossEntropyLoss()
            # loss_fct = FocalLoss(class_num=self.trigger_num_labels)
            loss_fct = BCEWithLogitsLoss(reduction="none")
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_trigger_start_logits = trigger_start_logits.view(-1, self.trigger_num_labels)
                active_trigger_end_logits = trigger_end_logits.view(-1, self.trigger_num_labels)

                active_trigger_start_labels = trigger_start_labels.view(-1, self.trigger_num_labels)
                active_trigger_end_labels = trigger_end_labels.view(-1, self.trigger_num_labels)

                trigger_start_loss = loss_fct(active_trigger_start_logits, active_trigger_start_labels.float())
                trigger_start_loss = trigger_start_loss * (active_loss.unsqueeze(-1))
                trigger_start_loss = torch.sum(trigger_start_loss)/torch.sum(active_loss)

                trigger_end_loss = loss_fct(active_trigger_end_logits, active_trigger_end_labels.float())
                trigger_end_loss = trigger_end_loss * (active_loss.unsqueeze(-1))
                trigger_end_loss = torch.sum(trigger_end_loss)/torch.sum(active_loss)

            else:
                trigger_start_loss = loss_fct(trigger_start_logits.view(-1, self.trigger_num_labels), trigger_start_labels.view(-1))
                trigger_end_loss = loss_fct(trigger_end_logits.view(-1, self.trigger_num_labels), trigger_end_labels.view(-1))
            trigger_loss = trigger_start_loss+ trigger_end_loss

        #######################################################
        ## role  
        # sequence_output_role = self.dropout(sequence_output)
        
        if self.config.with_gate:  
            global_logits = self.global_classifier(sequence_output)
            global_logits = self.activation(global_logits)
            role_start_logits = self.role_start_classifier(sequence_output) * global_logits
            role_end_logits = self.role_end_classifier(sequence_output) * global_logits
        else: 
            role_start_logits = self.role_start_classifier(sequence_output)
            role_end_logits = self.role_end_classifier(sequence_output)

        if role_start_labels is not None and role_end_labels is not None:
            # loss_fct = CrossEntropyLoss()
            # loss_fct = FocalLoss(class_num=self.role_num_labels)
            loss_fct = BCEWithLogitsLoss(reduction="none")
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_role_start_logits = role_start_logits.view(-1, self.role_num_labels)
                active_role_end_logits = role_end_logits.view(-1, self.role_num_labels)

                active_role_start_labels = role_start_labels.view(-1, self.role_num_labels)
                active_role_end_labels = role_end_labels.view(-1, self.role_num_labels)

                role_start_loss = loss_fct(active_role_start_logits, active_role_start_labels.float())
                role_start_loss = role_start_loss * (active_loss.unsqueeze(-1))
                role_start_loss = torch.sum(role_start_loss)/torch.sum(active_loss)

                role_end_loss = loss_fct(active_role_end_logits, active_role_end_labels.float())
                role_end_loss = role_end_loss * (active_loss.unsqueeze(-1))
                role_end_loss = torch.sum(role_end_loss)/torch.sum(active_loss)

            else:
                role_start_loss = loss_fct(role_start_logits.view(-1, self.role_num_labels), role_start_labels.view(-1))
                role_end_loss = loss_fct(role_end_logits.view(-1, self.role_num_labels), role_end_labels.view(-1))
            role_loss = role_start_loss+ role_end_loss
        
        outputs = (trigger_loss+ role_loss,) + ([trigger_start_logits, trigger_end_logits],) + ([role_start_logits, role_end_logits],) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)
        
    def predict_trigger(self, sequence_output):
        #######################################################
        ## trigger
        # sequence_output_trigger = self.dropout(sequence_output)
        trigger_start_logits = self.trigger_start_classifier(sequence_output)
        trigger_end_logits = self.trigger_end_classifier(sequence_output)

        threshold = 0.5
        trigger_start_logits = torch.sigmoid(trigger_start_logits)> threshold # 1498*256*217
        trigger_end_logits = torch.sigmoid(trigger_end_logits) > threshold
        # 64*256*65
        trigger_list = get_entities(trigger_start_logits.cpu().numpy(), trigger_end_logits.cpu().numpy())[0]
        return trigger_list

    def predict_role(self, sequence_output):     
        role_start_logits = self.role_start_classifier(sequence_output)
        role_end_logits = self.role_end_classifier(sequence_output)
        
        threshold = 0.5
        role_start_logits = torch.sigmoid(role_start_logits)> threshold # 1498*256*217
        role_end_logits = torch.sigmoid(role_end_logits) > threshold

        role_list = get_entities(role_start_logits.cpu().numpy(), role_end_logits.cpu().numpy())[0]
        return role_list

    def predict(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        outputs = outputs[2:]

        trigger_list = self.predict_trigger(sequence_output)
        role_list = self.predict_role(sequence_output)
        return trigger_list, role_list  # (loss), scores, (hidden_states), (attentions)

class BertForTokenBinaryClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.start_classifier = nn.Linear(config.hidden_size, config.num_labels) # batch_size, max_seq_length, hidden_size -> batch_size, max_seq_length, num_labels
        self.end_classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_labels=None, # batch* num_class* seq_length
        end_labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        
        start_logits = self.start_classifier(sequence_output)
        end_logits = self.end_classifier(sequence_output)

        outputs = ([start_logits, end_logits],) + outputs[2:]  # add hidden states and attention if they are here
        if start_labels is not None and end_labels is not None:
            # loss_fct = CrossEntropyLoss()
            # loss_fct = FocalLoss(class_num=self.num_labels)
            loss_fct = BCEWithLogitsLoss(reduction="none")
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_start_logits = start_logits.view(-1, self.num_labels)
                active_end_logits = end_logits.view(-1, self.num_labels)

                active_start_labels = start_labels.view(-1, self.num_labels)
                active_end_labels = end_labels.view(-1, self.num_labels)
                # attention_mask: 
                # ignore_index: [cls], [sep]
                # non_index: no label

                # print(active_loss, active_loss.shape, \
                #      active_logits,active_logits.shape,\
                #      active_labels,active_labels.shape,\
                #      labels, labels.shape)
                #2048 2048*435 2048 8*256 
                start_loss = loss_fct(active_start_logits, active_start_labels.float())
                active_loss = active_loss.float()
                start_loss = start_loss * (active_loss.unsqueeze(-1))
                start_loss = torch.sum(start_loss)/torch.sum(active_loss)

                end_loss = loss_fct(active_end_logits, active_end_labels.float())
                end_loss = end_loss * (active_loss.unsqueeze(-1))
                end_loss = torch.sum(end_loss)/torch.sum(active_loss)

            else:
                start_loss = loss_fct(start_logits.view(-1, self.num_labels), start_labels.view(-1))
                end_loss = loss_fct(end_logits.view(-1, self.num_labels), end_labels.view(-1))
            outputs = (start_loss+ end_loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)

class BertForTokenBinaryClassificationWithGate(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.config.with_gate:  
            # # 判断 当前 句子 是否存在 某种类型的实体：苏剑林 (使用 cls_embedding)
            # self.global_classifier = nn.Linear(config.hidden_size, config.num_labels) # batch_size, hidden_size -> batch_size, num_labels
            # 判断 当前 token 是否是 实体（任意类型）：whou
            self.global_classifier = nn.Linear(config.hidden_size, 1) # batch_size, max_seq_length, hidden_size -> batch_size, max_seq_length, 1

            self.activation = nn.ReLU()
            # 激活函数
            # 0-1 激活 
            # 梯度不反向传播
            # global_logits[global_logits < 0] = 0
            # global_logits[global_logits > 0] = 1
            # 梯度不反向传播
            # global_logits = torch.relu(torch.sign(global_logits))

            # relu 激活
            # 收敛速度快，效果好

            # sigmoid 激活

        self.start_classifier = nn.Linear(config.hidden_size, config.num_labels) # batch_size, max_seq_length, hidden_size -> batch_size, max_seq_length, num_labels
        self.end_classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_labels=None, # batch* num_class* seq_length
        end_labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        
        if self.config.with_gate:  
            # # 苏剑林
            # global_logits = self.global_classifier(sequence_output[:,0,:]).unsqueeze(-2)
            # whou
            global_logits = self.global_classifier(sequence_output)

            global_logits = self.activation(global_logits)

            start_logits = self.start_classifier(sequence_output) * global_logits
            end_logits = self.end_classifier(sequence_output) * global_logits
        else:
            start_logits = self.start_classifier(sequence_output)
            end_logits = self.end_classifier(sequence_output)

        outputs = ([start_logits, end_logits],) + outputs[2:]  # add hidden states and attention if they are here
        if start_labels is not None and end_labels is not None:
            # loss_fct = CrossEntropyLoss()
            # loss_fct = FocalLoss(class_num=self.num_labels)
            loss_fct = BCEWithLogitsLoss(reduction="none")
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_start_logits = start_logits.view(-1, self.num_labels)
                active_end_logits = end_logits.view(-1, self.num_labels)

                active_start_labels = start_labels.view(-1, self.num_labels)
                active_end_labels = end_labels.view(-1, self.num_labels)
                # attention_mask: 
                # ignore_index: [cls], [sep]
                # non_index: no label

                # print(active_loss, active_loss.shape, \
                #      active_logits,active_logits.shape,\
                #      active_labels,active_labels.shape,\
                #      labels, labels.shape)
                #2048 2048*435 2048 8*256 
                start_loss = loss_fct(active_start_logits, active_start_labels.float())
                active_loss = active_loss.float()
                start_loss = start_loss * (active_loss.unsqueeze(-1))
                start_loss = torch.sum(start_loss)/torch.sum(active_loss)

                end_loss = loss_fct(active_end_logits, active_end_labels.float())
                end_loss = end_loss * (active_loss.unsqueeze(-1))
                end_loss = torch.sum(end_loss)/torch.sum(active_loss)

            else:
                start_loss = loss_fct(start_logits.view(-1, self.num_labels), start_labels.view(-1))
                end_loss = loss_fct(end_logits.view(-1, self.num_labels), end_labels.view(-1))
            outputs = (start_loss+ end_loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)

class BertForTokenBinaryClassificationWithTriggerAndEventType(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.start_classifier = nn.Linear(config.hidden_size + 65, config.num_labels)
        self.end_classifier = nn.Linear(config.hidden_size + 65, config.num_labels)

        self.init_weights()

    # @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_labels=None, # batch* num_class* seq_length
        end_labels=None,
        event_type=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=torch.zeros_like(token_type_ids),
            # token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        # add triggrt embedding
        for i in range(sequence_output.size(0)):
            trigger_output=[]
            for j in range(sequence_output.size(1)):
                if token_type_ids[i][j]:
                    trigger_output.append(sequence_output[i][j])
            if trigger_output==[]: 
                print("segment_id == none")
                continue
            trigger_output = torch.stack(trigger_output,dim=0)
            trigger_output = torch.mean(trigger_output,dim=0)
            sequence_output[i] += trigger_output
        
        ## add event_type
        num_classes = 65
        event_type =  F.one_hot(event_type, num_classes=num_classes).float()
        seq_length = sequence_output.size(1)
        event_type = event_type.unsqueeze(1).repeat(1,seq_length,1)
        sequence_output =  torch.cat([sequence_output, event_type], axis=-1)


        sequence_output = self.dropout(sequence_output)

        start_logits = self.start_classifier(sequence_output)
        end_logits = self.end_classifier(sequence_output)

        outputs = ([start_logits, end_logits],) + outputs[2:]  # add hidden states and attention if they are here
        if start_labels is not None and end_labels is not None:
            # loss_fct = CrossEntropyLoss()
            # loss_fct = FocalLoss(class_num=self.num_labels)
            loss_fct = BCEWithLogitsLoss(reduction="none")
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_start_logits = start_logits.view(-1, self.num_labels)
                active_end_logits = end_logits.view(-1, self.num_labels)

                active_start_labels = start_labels.view(-1, self.num_labels)
                active_end_labels = end_labels.view(-1, self.num_labels)
                # attention_mask: 
                # ignore_index: [cls], [sep]
                # non_index: no label

                # print(active_loss, active_loss.shape, \
                #      active_logits,active_logits.shape,\
                #      active_labels,active_labels.shape,\
                #      labels, labels.shape)
                #2048 2048*435 2048 8*256 
                start_loss = loss_fct(active_start_logits, active_start_labels.float())
                start_loss = start_loss * (active_loss.unsqueeze(-1))
                start_loss = torch.sum(start_loss)/torch.sum(active_loss)

                end_loss = loss_fct(active_end_logits, active_end_labels.float())
                end_loss = end_loss * (active_loss.unsqueeze(-1))
                end_loss = torch.sum(end_loss)/torch.sum(active_loss)


            else:
                start_loss = loss_fct(start_logits.view(-1, self.num_labels), start_labels.view(-1))
                end_loss = loss_fct(end_logits.view(-1, self.num_labels), end_labels.view(-1))
            outputs = (start_loss+ end_loss,) + () + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)

