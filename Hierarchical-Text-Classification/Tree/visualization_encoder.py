from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import re

# 加载分词器和模型
tokenizer = BertTokenizer.from_pretrained('../Chinese-MentalBERT')
model = BertModel.from_pretrained('../Chinese-MentalBERT', output_attentions=True)

# 定义输入文本
text = "我也患有相当严重的社交焦虑症，从来没有走出家门去见我仅有的几个朋友"
# 去除标点符号
text = re.sub(r'[^\w\s]', '', text)

# 将输入文本分词
encoded_input = tokenizer(text, return_tensors='pt')

# 获取模型输出
output = model(**encoded_input)

# 提取注意力权重
attention = output.attentions

# 将注意力权重转换为NumPy数组，并在所有注意力头中进行平均
# 对每一层的注意力头进行平均，形状变为 (num_layers, seq_length, seq_length)
attention_weights = np.mean([att[0].detach().numpy() for att in attention], axis=1)

# 对每个token的注意力权重在所有层中进行平均
# 最终形状为 (seq_length, seq_length)
token_weights = attention_weights[-1,:,:]

# 聚焦于每个token从其他token接收到的注意力权重，并取平均值
# 得到的token_weights形状为 (seq_length,)
token_weights = token_weights.mean(axis=0)

# 获取分词后的tokens
tokens = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])

# 创建一个字典来存储tokens及其对应的权重
token_weights_dict = {token: weight for token, weight in zip(tokens, token_weights)}

# 排除特殊tokens [CLS] 和 [SEP]
token_weights_dict = {token: weight for token, weight in token_weights_dict.items() if token not in ["[CLS]", "[SEP]"]}

# 打印每个字及其对应的权重
print("Characters and their attention weights:")
for char, weight in token_weights_dict.items():
    print(f"Character: {char}, Weight: {weight}")

# 按权重降序排序，并选择前5个字
sorted_chars = sorted(token_weights_dict.items(), key=lambda item: item[1], reverse=True)[:7]

# 打印权重最高的前5个字
print("\nTop 5 characters with the highest attention weights:")
for char, weight in sorted_chars:
    print(f"Character: {char}, Weight: {weight}")

