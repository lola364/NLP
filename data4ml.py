import re
from collections import defaultdict, Counter
import math
import json

def read_label_list(path):
    """
    从文件中读取类别标签列表。
    :param path: 标签列表文件路径
    :return: 类别名称列表
    """
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]


def read_dataset(path, label_list):
    """
    读取数据集并将多标签转为二值向量。
    :param path: 数据集路径
    :param label_list: 所有类别列表
    :return: inputs (文本列表), labels (多标签二值矩阵)
    """
    inputs = []
    labels = []
    label_to_index = {label: i for i, label in enumerate(label_list)}  # 构建类别到索引的映射

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            inputs.append(sample['text'])

            # 初始化标签向量
            label_vector = [0] * len(label_list)
            for label in sample['label']:
                if label in label_to_index:
                    label_vector[label_to_index[label]] = 1
            labels.append(label_vector)

    return inputs, labels
# 数据预处理
class MyDataset:
    """
    文本数据预处理类，包括停用词加载、词典构建和文本向量化功能。
    """

    def __init__(self):
        self.vocab = {}
        self.stop_words = []

    def set_stopword(self, path='data/en_stopwords.txt'):
        """
        加载停用词表。
        :param path: 停用词文件路径
        """
        with open(path, 'r', encoding='utf-8') as fr:
            self.stop_words = [line.strip().lower() for line in fr.readlines()]

    def preprocess_text(self, text):
        """
        对文本进行预处理：移除标点符号、转换为小写、分词。
        :param text: 输入的原始文本
        :return: 处理后的单词列表
        """
        text = re.sub(r'[^\w\s]', '', text)  # 移除标点符号
        text = text.lower()  # 转为小写
        return text.split()  # 按空格分词

    def build_vocab(self, inputs, max_size=5000, min_freg=1):
        """
        基于词频构建词典。
        :param inputs: 文本列表
        :param max_size: 词典最大容量
        :param min_freg: 最小词频
        """
        cnt = {}
        for data in inputs:
            words = self.preprocess_text(data)
            for word in words:
                if word not in cnt:
                    cnt[word] = 1
                else:
                    cnt[word] += 1
        # 迭代cnt.item()中的每个元素，若词频>=1且这个词不在停用词表里，就放入列表，根据词频升序排列
        cnt = sorted(
            [item for item in cnt.items() if item[1] >= min_freg and item[0] not in self.stop_words],
            key=lambda x: x[1], reverse=True
        )
        # 文本出现无法读到的值，使用一个特殊符号填充。
        self.vocab['<pad>'] = 0
        for i, (word, _) in enumerate(cnt[:max_size], 1):
            self.vocab[word] = i

    def build_tfidf_vocab(self, inputs, max_size=5000):
        """
        基于 TF-IDF 方法构建词典。
        :param inputs: 文本列表
        :param max_size: 词典最大容量
        """
        df = defaultdict(int)
        tf = []
        total_docs = len(inputs)

        for doc in inputs:
            doc_tf = defaultdict(int)
            words = self.preprocess_text(doc)
            for word in words:
                doc_tf[word] += 1
            for word in set(words):
                df[word] += 1
            tf.append(doc_tf)

        tfidf_scores = defaultdict(float)
        for doc_tf in tf:
            for word, freq in doc_tf.items():
                tf_val = freq / sum(doc_tf.values())
                idf_val = math.log(total_docs / (1 + df[word]))
                tfidf_scores[word] += tf_val * idf_val

        sorted_tfidf = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)

        self.vocab = {'<pad>': 0}
        for i, (word, _) in enumerate(sorted_tfidf[:max_size], 1):
            self.vocab[word] = i

    def transform(self, inputs, flag=0):
        """
        将文本数据转换为向量。
        :param inputs: 文本列表
        :param flag: 0 表示词集模型，1 表示词袋模型
        :return: 转换后的向量
        """
        samples = [] #创建一个空列表 samples 用于存储每个文本对应的向量。
        for doc in inputs:
            words = self.preprocess_text(doc)
            if flag == 0:
                word_set = set(words)  #将单词列表转为集合，保留唯一单词。
                sample = [1 if word in word_set else 0 for word in self.vocab]
            elif flag == 1:
                sample = [0] * len(self.vocab)   # 初始化向量，长度等于词典大小，值全为 0
                word_count = Counter(words)   #统计文本中每个单词的出现次数
                for word, freq in word_count.items():
                    if word in self.vocab:
                        sample[self.vocab[word]] = freq
            samples.append(sample)
        return samples


if __name__ == '__main__':
    label_list_file = 'data/label_list.txt'
    label_list = read_label_list(label_list_file)
    print(f"Loaded {len(label_list)} labels: {label_list}")

    # 读取训练集和验证集
    train_inputs, train_labels = read_dataset('data/train.json',label_list)
    valid_inputs, valid_labels = read_dataset('data/valid.json',label_list)

    # 检查第一个样本的文本和标签
    print(f"First sample text: {train_inputs[0]}")
    print(f"First sample labels: {train_labels[0]}")
    # 实例化数据处理类
    ds = MyDataset()
    ds.set_stopword('data/en_stopwords.txt')  # 停用词文件路径

    # 构建 TF-IDF 词典
    ds.build_tfidf_vocab(train_inputs, max_size=5000)

    # 转换前 5 个样本为向量表示
    sample_vectors = ds.transform(train_inputs[:5])
    # 使用 transform 方法将前 5 个样本转换为向量。打印转换后的向量，检查是否符合预期格式（数值向量化结果）。
    print("First 5 sample vectors:")
    print(sample_vectors)

    # 检查对应的标签（多标签二值矩阵）
    print("First 5 sample labels (binary matrix):")
    print(train_labels[:5])
