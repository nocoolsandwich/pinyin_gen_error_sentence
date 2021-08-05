# SoftMaskedBert
```
每条句子中15%的token其中：
    80%token谐音下频率前50%词替换
    20%token随机频率前50%词替换
    替换的位置label为1
    使用label训练gru
    使用mlm训练bert
    loss加权一下
```
Soft-Masked Bert 复现论文:https://arxiv.org/pdf/2005.07421.pdf
