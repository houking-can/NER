# A simple BiLSTM-CRF model for Chinese Named Entity Recognition


### data

如果有人工标注的数据，或者想尝试一下使用词典标注的数据训练CRF，按照data目录下准备训练数据。

- 预处理好的数据, `train`and `test` ，格式如下：
```
中	B-LOC
国	I-LOC
很	O
大	O

句	O
子	O
结	O
束	O
是	O
空	O
行	O

```
- 字典 `word2id.pkl` ，对每个字映射到唯一id，如： `的：0`  

- 预训练的 `embedding.npy`，如果没有这个文件，就会使用随机初始化的词向量。对于词向量，建议使用全部数据自己训练，训练完后能同时得到word2id和embedding。

- raw存放原始数据，需要你们自己写处理脚本，转成训练数据的BIO形式。




### dependency

- tensorflow 1.2
- python3
- perl

## How to Run
train: `python main.py --mode=train `
test :`python main.py --mode=test --demo_model=1521112368`
demo:  `python main.py --mode=demo --demo_model=1521112368`
predict: `python main.py --mode=predict --demo_model=1521112368 --predict_path='/home/...'`
predict模式是针对大量需要做预测数据的情况

# Thanks

[https://github.com/Determined22/zh-NER-TF](https://github.com/Determined22/zh-NER-TF)

这个repo是基于上面这个项目进行修改，优化了部分代码。