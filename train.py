# Import Gpt2- primitives
# Load fineweb 5B or something smaller
# Adjust future lookers, stop gradients, + 4 projections into future :) 
# load weights
# adjust loss function
from datasets import load_dataset
from torch.utils.data import DataLoader
from datasets import Dataset
import wandb
import torch
from tqdm import tqdm
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Optional, Tuple, List
from dataclasses import dataclass
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

@dataclass
class CausalLMOutputWithCrossAttentionsAndFuture(CausalLMOutputWithCrossAttentions):
    future_logits: torch.Tensor = None
    losses: List[torch.Tensor] = None


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  

FUTURE_LOOK = 3
class FutureProbeGPT2(GPT2LMHeadModel):
    def __init__(self, *args, **kwargs):
        super(FutureProbeGPT2, self).__init__(*args, **kwargs)
        # model_dim 
        model_dim = self.config.n_embd
        self.oracle = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, 3*model_dim),
        )
        
        # disable gradients for lm_head
        for param in self.lm_head.parameters():
            param.requires_grad = False
        
    
    def forward(self,  
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ):
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            transformer_outputs = self.transformer(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = transformer_outputs[0]

            # Set device for model parallelism
            if self.model_parallel:
                torch.cuda.set_device(self.transformer.first_device)
                hidden_states = hidden_states.to(self.lm_head.weight.device)

            # lm_logits = self.lm_head(hidden_states)
            # future projection
            future_logits = self.oracle((hidden_states.detach()))  # or torch.randn_like for blind tests
            # chunk the future logits
            future_logits = future_logits.chunk(FUTURE_LOOK, dim=-1)
            future_logits = [f + hidden_states.detach() for f in future_logits]
            # stack the future logits
            future_logits = torch.stack(future_logits, dim=1)
            
            hidden_states = torch.cat([hidden_states[:, None, ...], future_logits], dim=1)

            # pass the future logits through the main head without grad
            lm_logits = self.lm_head(hidden_states) # (B, FUTURE_LOOK, S, V)
            
            # combine into one tensor
      
            loss = None
            losses = []
            if labels is not None:
                
                for i in range(lm_logits.size(1)):
                    shift_logits = lm_logits[:, i, :-(1 + i), :].contiguous()
                    shift_labels = labels[..., (1 + i):].contiguous()
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    losses.append(loss)

                loss = torch.stack(losses).mean()
               

            if not return_dict:
                output = (lm_logits[:, 0],) + transformer_outputs[1:]
                return ((loss,) + output) if loss is not None else output

            return CausalLMOutputWithCrossAttentionsAndFuture(
                loss=loss,
                losses=losses,
                logits=lm_logits[:, 0],
                future_logits=lm_logits[:, 1:],
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
                cross_attentions=transformer_outputs.cross_attentions,
            )

def setup_model():
    model = FutureProbeGPT2.from_pretrained('gpt2')
    return model


def setup_optimizer(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    return optimizer

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=512,  # Adjust as needed
        padding='max_length',
        return_tensors='pt'
    )


def set_data_loader(batch_size, num_workers):
    ds = load_dataset("VisionTheta/fineweb-1B", split='train')  # adjust size as needed
    ds = ds.map(
        tokenize_function,
        batched=True,
        batch_size=1000,  # Larger batch size for tokenization
        num_proc=32,  # Use multiple processes
        remove_columns=ds.column_names  # Remove unnecessary columns
    )
    ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # Parallel data loading
        pin_memory=True  # Faster data transfer to GPU
    )
    return dataloader

OPTIMIZATION_STEPS = 1_000_000

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.init(project='future-probe-gpt2')
    model = setup_model()
    model = model.type(torch.bfloat16).to(device)
    optimizer = setup_optimizer(model)
    dataloader = set_data_loader(batch_size=16, num_workers=2)
    model.train()
    # prgbar = tqdm(dataloader)
    global_step = 0
    prgbar = tqdm(total=OPTIMIZATION_STEPS, desc='Optimizing... :3')
    while global_step < OPTIMIZATION_STEPS:
        for batch in (dataloader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            # print(input_ids.shape)
            # break;
            attention_mask = batch['attention_mask'].type(torch.bfloat16).to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            losses = outputs.losses
            loss.backward()
            optimizer.step()
            info = {}
            for i, l in enumerate(losses):
                info[f'loss_{i}'] = l.item()
                info[f'perplexity_{i}'] = torch.exp(l).item()

            prgbar.set_postfix(losses=[f"{l.item():.2f}" for l in losses])
            wandb.log(info)
            global_step += 1
            prgbar.update(1)
            if global_step % 1000 == 0:
                torch.save(model.state_dict(), 'model.pth')
            if global_step >= OPTIMIZATION_STEPS:
                break



if __name__ == '__main__':
    train()
