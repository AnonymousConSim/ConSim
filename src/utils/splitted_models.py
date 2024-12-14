from abc import ABC, abstractmethod

from tqdm import tqdm
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
import transformers


class SplittedBert(transformers.BertForSequenceClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, **kwargs):
        outputs = super().forward(**kwargs)
        return outputs.logits  

    def features(self, **inputs):
        outputs = self.bert(**inputs)
        return self.dropout(outputs[1])
    
    def end_model(self, x):
        x = self.classifier(x)
        return x
    
    def del_base_model(self):
        del self.bert
        del self.dropout


class SplittedDeberta(transformers.DebertaV2ForSequenceClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, **kwargs):
        outputs = super().forward(**kwargs)
        return outputs.logits

    def features(self, **kwargs):
        outputs = (self.deberta(**kwargs)).last_hidden_state
        return self.pooler(outputs)
    
    def end_model(self, x):
        x = self.classifier(x)
        x = self.dropout(x)
        return x
    
    def del_base_model(self):
        del self.deberta
        del self.pooler


class SplittedDistilBert(transformers.DistilBertForSequenceClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, **kwargs):
        outputs = super().forward(**kwargs)
        return outputs["logits"]
    
    def base(self, **kwargs):
        return self.distilbert(**kwargs)

    def features_processing(self, base_outputs):
        return base_outputs[0][:, 0]

    def features(self, **kwargs):
        base_outputs = self.base(**kwargs)
        return self.features_processing(base_outputs)

    def end_model(self, x):
        x = self.pre_classifier(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
    
    def end_model_to(self, device):
        self.pre_classifier.to(device)
        self.dropout.to(device)
        self.classifier.to(device)
    
    def del_base_model(self):
        del self.distilbert


class PositiveDistilBert(SplittedDistilBert):
    def features_processing(self, base_outputs):
        pooled_outputs =  base_outputs[0][:, 0]
        return nn.ReLU()(pooled_outputs)
    
    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                labels: Optional[torch.LongTensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
            ) -> Union[transformers.modeling_outputs.SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        base_outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        positive_pooled_output = self.features_processing(base_outputs)

        logits = self.end_model(positive_pooled_output)

        # print("DEBUG: splitted_models: PositiveDistilBert: forward: labels", labels.shape, "\n", labels)
        # print("DEBUG: splitted_models: PositiveDistilBert: forward: logits", logits.shape, "\n", logits)

        loss = None
        if labels is not None and not (labels < 0).any():
            # safety net for bug with cola dataset
            if (labels < 0).any():
                print("WARNING: Negative labels found in PositiveDistilBert forward.")
                exit()

            if self.config.num_labels == 1:
                #  We are doing regression
                loss = nn.MSELoss()(logits.view(-1), labels.view(-1))
            elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                loss = nn.CrossEntropyLoss()(logits.view(-1, self.config.num_labels), labels.view(-1))
            else:
                loss = nn.BCEWithLogitsLoss()(logits, labels)

        
        if not return_dict:
            output = (logits,) + base_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return transformers.modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=base_outputs.hidden_states,
            attentions=base_outputs.hidden_states,
        )
    
    def del_base_model(self):
        del self.distilbert
    

class SplittedRoberta(transformers.RobertaForSequenceClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, **kwargs):
        outputs = super().forward(**kwargs)
        return outputs.logits  

    def features(self, **kwargs):
        outputs = self.roberta(**kwargs)
        return (outputs.last_hidden_state)[:,0,:]
    
    def end_model(self, x):
        x = self.classifier.dense(x)
        x = self.classifier.dropout(x)
        x = self.classifier.out_proj(x)
        return x
    
    def del_base_model(self):
        del self.roberta


class SplittedT5(transformers.T5ForSequenceClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def base(self, **kwargs):
        if ("decoder_input_ids" not in kwargs or kwargs["decoder_input_ids"] is None)\
            and ("decoder_inputs_embeds" not in kwargs or kwargs["decoder_inputs_embeds"] is None):
            kwargs["decoder_input_ids"] = self._shift_right(kwargs["input_ids"])
        return self.transformer(**kwargs)

    def features_processing(self, input_ids, base_outputs):
        sequence_output = base_outputs[0]

        eos_mask = input_ids.eq(self.config.eos_token_id).to(sequence_output.device)

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        batch_size, _, hidden_size = sequence_output.shape
        sentence_representation = sequence_output[eos_mask, :].view(batch_size, -1, hidden_size)[:, -1, :]
        return sentence_representation
    
    def features(self, **kwargs):
        base_outputs = self.base(**kwargs)
        return self.features_processing(kwargs["input_ids"], base_outputs)
    
    def end_model(self, x):
        x = self.classification_head(x)
        return x
    
    def end_model_to(self, device):
        self.classification_head.to(device)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[transformers.cache_utils.Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, transformers.modeling_outputs.SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        base_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sentence_representation = self.features_processing(input_ids, base_outputs)

        logits = self.end_model(sentence_representation)
        
        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                #  We are doing regression
                loss = nn.MSELoss()(logits.squeeze(), labels.squeeze())
            elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                loss = nn.CrossEntropyLoss()(logits.view(-1, self.config.num_labels), labels.view(-1))
            else:
                loss = nn.BCEWithLogitsLoss()(logits, labels)
        
        if not return_dict:
            output = (logits,) + base_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return transformers.modeling_outputs.SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=base_outputs.past_key_values,
            hidden_states=base_outputs.hidden_states,
            attentions=base_outputs.attentions,
        )
    
    def del_base_model(self):
        del self.transformer


class PositiveT5(SplittedT5):
    def features_processing(self, input_ids, base_outputs):
        sentence_representation = super().features_processing(input_ids, base_outputs)
        return nn.ReLU()(sentence_representation)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, transformers.modeling_outputs.Seq2SeqSequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        # Copied from models.bart.modeling_bart.BartModel.forward different to other models, T5 automatically creates
        # decoder_input_ids from input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )
            decoder_input_ids = self._shift_right(input_ids)

        base_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sentence_representation = self.features_processing(input_ids, base_outputs)

        logits = self.end_model(sentence_representation)
        
        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                #  We are doing regression
                loss = nn.MSELoss()(logits.squeeze(), labels.squeeze())
            elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                loss = nn.CrossEntropyLoss()(logits.view(-1, self.config.num_labels), labels.view(-1))
            else:
                loss = nn.BCEWithLogitsLoss()(logits, labels)

        
        if not return_dict:
            output = (logits,) + base_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return transformers.modeling_outputs.Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=base_outputs.past_key_values,
            decoder_hidden_states=base_outputs.decoder_hidden_states,
            decoder_attentions=base_outputs.decoder_attentions,
            cross_attentions=base_outputs.cross_attentions,
            encoder_last_hidden_state=base_outputs.encoder_last_hidden_state,
            encoder_hidden_states=base_outputs.encoder_hidden_states,
            encoder_attentions=base_outputs.encoder_attentions,
        )
    
    def del_base_model(self):
        del self.transformer


datasets_system_prompts = {
    "BIOS": """
You are a classifier model the BIOS dataset. The dataset consist of the 28 classes:
['surgeon', 'pastor', 'photographer', 'professor', 'chiropractor', 'software_engineer', 'teacher',
'poet', 'dj', 'rapper', 'paralegal', 'physician', 'journalist', 'architect', 'attorney',
'yoga_teacher', 'nurse', 'painter', 'model', 'composer', 'personal_trainer', 'filmmaker',
'comedian', 'accountant', 'interior_designer', 'dentist', 'psychologist', 'dietitian']
You will be given a biography and you should return the occupation of the person described in the biography.
The occupation should belong to the occupations above, no other occupation will be accepted.
You should only predict the occupation and nothing else.""",
    "BIOS_ng": """
You are a classifier model the BIOS dataset. The dataset consist of the 28 classes:
['surgeon', 'pastor', 'photographer', 'professor', 'chiropractor', 'software_engineer', 'teacher',
'poet', 'dj', 'rapper', 'paralegal', 'physician', 'journalist', 'architect', 'attorney',
'yoga_teacher', 'nurse', 'painter', 'model', 'composer', 'personal_trainer', 'filmmaker',
'comedian', 'accountant', 'interior_designer', 'dentist', 'psychologist', 'dietitian']
You will be given a biography and you should return the occupation of the person described in the biography.
The occupation should belong to the occupations above, no other occupation will be accepted.
You should only predict the occupation and nothing else.""",
    "BIOS10": """
You are a classifier model on a subset of the BIOS dataset. The dataset consist of the 10 most frequent classes:
['surgeon', 'photographer', 'professor', 'teacher', 'physician', 'journalist', 'attorney', 'nurse', 'dentist', 'psychologist']
You will be given a biography and you should return the occupation of the person described in the biography.
The occupation should belong to the occupations above, no other occupation will be accepted.
You should only predict the occupation and nothing else.""",
    "IMDB": """
You are a classifier model on the IMDB dataset. The dataset consist of the 2 classes:
['pos', 'neg']
You will be given a review and you should return the sentiment of the review.
The sentiment should belong to the classes above, no other sentiment will be accepted.
You should only predict the sentiment and nothing else.""",
    "tweet_eval_emotion": """
You are a classifier model on the TweetEval dataset for emotion analysis. The dataset consist of the 4 classes:
['anger', 'joy', 'optimism', 'sadness']
You will be given a tweet and you should return the emotion of the tweet.
The emotion should belong to the classes above, no other emotion will be accepted.
You should only predict the emotion and nothing else.""",
    "rotten_tomatoes": """
You are a classifier model on the Rotten Tomatoes dataset. The dataset consist of the 2 classes:
['pos', 'neg']
You will be given a review and you should return the sentiment of the review.
The sentiment should belong to the classes above, no other sentiment will be accepted.
You should only predict the sentiment and nothing else.""",
}


class SplittedLlamaForCausalLM(transformers.LlamaForCausalLM):
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.end_model_batch_size = 1024
        self.tqdm = False
    
    def setup(self, tokenizer, dataset_config):
        self.tokenizer = tokenizer
        self.prompt_completion = self.tokenizer(["<|start_header_id|>assistant<|end_header_id|>\n\n"], return_tensors="pt").input_ids

        self.dataset = dataset_config.name
        assert self.dataset in datasets_system_prompts, f"Dataset {self.dataset} not found in datasets_system_prompts."
        self.num_labels = dataset_config.num_labels

        # list of tokens id for each class (the first token of the prediction represent the class)
        self.classes_tokens = torch.zeros(self.num_labels)
        for i, label in enumerate(dataset_config.labels_names):
            self.classes_tokens[i] = self.tokenizer(label).input_ids[1]

        self.system_prompt = datasets_system_prompts[self.dataset]

        self.make_prompt_from_input = lambda text: [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text},
        ]

    def input_ids_from_text(self, text: str):
        # apply prompting
        prompt = self.make_prompt_from_input(text)

        # tokenize
        input_ids = self.tokenizer.apply_chat_template(prompt, add_general_prompt=True, return_tensors="pt")

        # add completion token '<|start_header_id|>assistant<|end_header_id|>\n\n
        input_ids = torch.cat([input_ids, self.prompt_completion], dim=1)
        return input_ids

    def features(self, inputs: List[str]):
        if isinstance(inputs, str):
            inputs = [inputs]
        
        embeddings = []
        for text in tqdm(inputs, disable=not self.tqdm):
            # tokenize
            input_ids = self.input_ids_from_text(text).to(self.device)

            # forward pass
            # (n, num_tokens, hidden_size) where n is 1
            last_hidden_state = self.model(input_ids=input_ids, return_dict=True)["last_hidden_state"]

            # (hidden_size,)
            next_token_embedding = last_hidden_state[0, -1].cpu()

            embeddings.append(next_token_embedding)

        embeddings = torch.stack(embeddings)

        return embeddings

    def __call__(self, inputs: List[str]):
        # compute embeddings
        embeddings = self.features(inputs)

        # batched call to end_model
        logits = torch.cat([
            self.end_model(embeddings[i:i+self.end_model_batch_size])
            for i in range(0, len(embeddings), self.end_model_batch_size)
        ])

        return logits

    def end_model(self, embeddings: torch.Tensor):
        # (n, num_tokens)
        tokens_logits = self.lm_head(embeddings).float()

        # (n, num_classes)
        logits = torch.index_select(tokens_logits, dim=1, index=self.classes_tokens.long())

        return logits
    
    def end_model_to(self, device):
        self.lm_head.to(device)  
        self.classes_tokens = self.classes_tokens.to(device)
    
    def del_base_model(self):
        del self.model
