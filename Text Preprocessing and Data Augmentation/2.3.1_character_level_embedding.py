import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# 示例文本数据，包含OOV词汇
sentences = [
    "深度学习是一种机器学习技术",
    "语言处理在人工智能中扮演重要角色",
    "自然语言处理是一门重要学科",
    "新兴技术包括量子计算和深度神经网络"
]


# 创建字符级别词汇表
def build_char_vocab(sentences):
    char_vocab = set()
    for sentence in sentences:
        for char in sentence:
            char_vocab.add(char)
    char2idx = {char: idx + 1 for idx, char in enumerate(char_vocab)}
    char2idx['<PAD>'] = 0  # 添加填充标记
    return char2idx


char2idx = build_char_vocab(sentences)
vocab_size = len(char2idx)


# 定义字符级嵌入模型
class CharLevelEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(CharLevelEmbedding, self).__init__()
        self.char_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

    def forward(self, x):
        embedded_chars = self.char_embedding(x)
        # 计算字符嵌入的平均表示
        return embedded_chars.mean(dim=1)


# 生成字符ID序列
def encode_sentence(sentence, char2idx, max_len=20):
    char_ids = [char2idx.get(char, 0) for char in sentence]  # 0作为OOV字符
    if len(char_ids) < max_len:
        char_ids += [0] * (max_len - len(char_ids))  # 填充字符
    else:
        char_ids = char_ids[:max_len]
    return torch.tensor(char_ids)


encoded_sentences = [encode_sentence(sentence, char2idx) for sentence in sentences]
encoded_tensor = torch.stack(encoded_sentences)

# 初始化模型
embed_dim = 16  # 嵌入维度
char_embed_model = CharLevelEmbedding(vocab_size, embed_dim)

# 生成字符级嵌入
char_embeddings = char_embed_model(encoded_tensor)

# 输出结果
print("字符级词汇表:", char2idx)
print("编码后的字符ID序列:\n", encoded_tensor)
print("字符级嵌入:\n", char_embeddings)
