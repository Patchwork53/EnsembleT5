# EnsembleT5 
### T5 models from huggingface/transformers
A simple class override that allows ensembling T5 models from huggingface/transformers during inference. Works with trainer().


### Usage
```
model1 = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
model2 = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
num_beams = 3
model = EnsembledT5ForConditionalGeneration(model1, model2, num_beams)
```

### Current Limitations
<ul>
<li> Currently works with two models only </li>
<li> Only for inference, using either model.generate() or trainer with the flag predict_with_generate = True </li>
<li> The models must be the same architecture with the same config files. Ideally sister checkpoints. </li>
<li> Currently tested only for beam_search generation </li>
<li> Works on single GPU only </li>
