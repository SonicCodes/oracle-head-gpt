# oracle-head-gpt

I've always had a feeling hidden states in LLMs are not just next token predictors as there's no actual constraint for them to not contain future trajectories, this information is destroyed/redacted by the discreetization or sampling procedure, 

to prove this I trained an oracle head that takes a detached hiddenstate and learn residuals to transform it into N+1, N+2, N+3 token's hiddenstates, Futhermore I froze the lm_head to make sure the model to make sure the oracle is not cheating, loss is computed normally with different shifting masks...

I started with open-ai's GPT2 113M checkpoint, used FineWeb-1B subsample with 512 ctx length, and did the training procedure on an A100 with batch size of 16, looking at the results it seems that the loss goes as expected! loss progresses N, N+1, N+2, N+3

==== loss screenshot ====