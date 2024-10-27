# oracle-head-gpt

I've always had a feeling hidden states in LLM's hidden states are not just next token predictors as there's no constraint for them to not contain future trajectories, this information is destroyed/redacted by the discreetization or sampling procedure, but it's kinda still there...

to prove this I trained an oracle head that takes a detached hiddenstate and learn residuals to transform it into N+1, N+2, N+3 token's hidden states, Futhermore I froze the lm_head to make sure the oracle is not cheating, loss is computed normally with different shifting masks per position...

I started with open-ai's GPT2 113M checkpoint, used FineWeb-1B subsample with 512 ctx length, and did the training procedure on an A100 with batch size of 16, looking at the results it seems that the loss goes as expected! loss becomes worse, as N, N+1, N+2, N+3

>> I have two versions of this, sighted and blind oracle
>> 1) Sighted oracle: uses previous hiddenstates to predict the residuals that transform it to the next tokens,
![image](https://github.com/user-attachments/assets/a9f17153-4665-4249-be9e-165a905b8860)

>> 2) Blind oracle: recieves noise to predict the next residuals that  transform it to the next tokens,
![image](https://github.com/user-attachments/assets/eae623c3-7801-4cd0-a5e8-289bf84c0133)


Reason for the blind test is to make sure it's not actually pattern matching as it doesn't have a context of what it's adding these positional residuals to, but we technically should want it to know as the embedding space is not as structured as typical contrasive ones go...

(I plan to work on this soon, this is a rushed reproduction of a previous work i did , but accidentally deleted)
