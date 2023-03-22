from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers import T5PreTrainedModel,T5ForConditionalGeneration
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from transformers.utils import ModelOutput
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import inspect

class EnsembledT5ForConditionalGeneration(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, model1: T5ForConditionalGeneration, model2: T5ForConditionalGeneration, num_beams):
        config = model1.config

        super().__init__(config)

        self.model1 = model1
        self.model2 = model2
        self.num_beams = num_beams #needed in the overriden expand_inputs_for_generation() fcn
        self.running_encoder_outputs = None
        # self.model_dim = config.d_model

        # self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # encoder_config = copy.deepcopy(config)
        # encoder_config.is_decoder = False
        # encoder_config.use_cache = False
        # encoder_config.is_encoder_decoder = False
        # self.encoder = T5Stack(encoder_config, self.shared)
        
        # decoder_config = copy.deepcopy(config)
        # decoder_config.is_decoder = True
        # decoder_config.is_encoder_decoder = False
        # decoder_config.num_layers = config.num_decoder_layers
        # self.decoder = T5Stack(decoder_config, self.shared)

        # self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        # self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        
    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5ForConditionalGeneration.parallelize` is deprecated and will be removed in v5 of Transformers, you"
            " should load your model with `device_map='balanced'` in the call to `from_pretrained`. You can also"
            " provide your own `device_map` but it needs to be a dictionary module_name to device, so for instance"
            " {'encoder.block.0': 0, 'encoder.block.1': 1, ...}",
            FutureWarning,
        )
        print("parallelize not implemented")

        
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        print("deparallelize not implemented")

    def get_input_embeddings(self):
        return self.model1.shared

    def set_input_embeddings(self, new_embeddings):
        self.model1.shared = new_embeddings
        self.model1.encoder.set_input_embeddings(new_embeddings)
        self.model1.decoder.set_input_embeddings(new_embeddings)
        self.model2.shared = new_embeddings
        self.model2.encoder.set_input_embeddings(new_embeddings)
        self.model2.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.model1.lm_head = new_embeddings
        self.model2.lm_head = new_embeddings

    def get_output_embeddings(self):
        print("needs work? get_output_embeddings")
        return self.model1.lm_head

    def get_encoder(self):
        # model.generate() calls get_encoder to encode the input sentence before handing off the result to beam_search
        return self.model1.encoder, self.model2.encoder

    def get_decoder(self):
        print("needs work? get_decoder")    
        return self.model1.decoder

    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,

        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        encoder_outputs2: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,

        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, T5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
#         print("-"*50)
# #         print(input_ids, flush = True)
# #         print(decoder_input_ids)
#         if encoder_outputs is not None:
#             print(encoder_outputs.last_hidden_state.shape)
#         else:
#             print("None")
#         if encoder_outputs2 is not None:
#             print(encoder_outputs2.last_hidden_state.shape)
#         else:
#             print("None")
#         print("-"*50, flush=True)
       
        res1 = self.model1(
            input_ids, 
            attention_mask, 
            decoder_input_ids,
            decoder_attention_mask,  
            head_mask,
            decoder_head_mask,
            cross_attn_head_mask,
            encoder_outputs,
            past_key_values,
            inputs_embeds,
            decoder_inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict
            )
        res2 = self.model2(
            input_ids, 
            attention_mask, 
            decoder_input_ids,
            decoder_attention_mask,
            
            head_mask,
            decoder_head_mask,
            cross_attn_head_mask,
            encoder_outputs2,
            
            past_key_values,
            inputs_embeds,
            decoder_inputs_embeds,
            labels,

            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict
            )
       

        loss = None
        if res1.loss is not None and res2.loss is not None:
            loss = res1.loss + res2.loss

        lm_logits = None
        if res1.logits is not None and res2.logits is not None:
            lm_logits = res1.logits + res2.logits


        past_key_values = res1.past_key_values
        decoder_hidden_states = res1.decoder_hidden_states
        decoder_attentions = res1.decoder_attentions
        cross_attentions = res1.cross_attentions
        encoder_last_hidden_state = res1.encoder_last_hidden_state
        encoder_hidden_states = res1.encoder_hidden_states
        encoder_attentions = res1.encoder_attentions

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=past_key_values,
            decoder_hidden_states=decoder_hidden_states,
            decoder_attentions=decoder_attentions,
            cross_attentions=cross_attentions,
            encoder_last_hidden_state=encoder_last_hidden_state,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attentions=encoder_attentions
        )
        # return res1

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        encoder_outputs2=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        
#         print(encoder_outputs.last_hidden_state.shape)
#         print(encoder_outputs2.last_hidden_state.shape)
#         print("inside prep input for gen", flush=True)
        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "encoder_outputs2": encoder_outputs2,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        self.model1._shift_right(labels)
        return self.model2._shift_right(labels)


    def _reorder_cache(self, past_key_values, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past
    
    
    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
        ) -> Dict[str, Any]:
        
        
        
        encoder,encoder2 = self.get_encoder()

        # 2. Prepare encoder args and encoder kwargs from model kwargs.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)
        model_kwargs["encoder_outputs2"]: ModelOutput = encoder2(**encoder_kwargs)    
            
        
        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if dict_to_expand[key] is not None and isinstance(dict_to_expand[key], torch.Tensor):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(self.num_beams, dim=0)  #using self.num_beams
            return dict_to_expand
        
        model_kwargs["encoder_outputs2"] = _expand_dict_for_generation(model_kwargs["encoder_outputs2"]) #encoder_outputs is expanded internally with _expand_inputs_for_generation, need to do this manually for encoder_outputs2
        

        return model_kwargs
    
    
# Tried to override this method but it interferes with another function of the same name in generation_utils.py
#     @staticmethod
#     def _expand_inputs_for_generation(
#         expand_size: int = 1,
#         is_encoder_decoder: bool = False,
#         input_ids: Optional[torch.LongTensor] = None,
#         **model_kwargs,
#     ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
#         """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""

#         def _expand_dict_for_generation(dict_to_expand):
#             for key in dict_to_expand:
#                 if dict_to_expand[key] is not None and isinstance(dict_to_expand[key], torch.Tensor):
#                     dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
#             return dict_to_expand

#         if input_ids is not None:
#             input_ids = input_ids.repeat_interleave(expand_size, dim=0)

#         model_kwargs = _expand_dict_for_generation(model_kwargs)

#         if is_encoder_decoder:
#             if model_kwargs.get("encoder_outputs") is None:
#                 raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
#             model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])
#             model_kwargs["encoder_outputs2"] = _expand_dict_for_generation(model_kwargs["encoder_outputs2"])

#         return input_ids, model_kwargs
