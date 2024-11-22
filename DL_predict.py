import numpy as np
import pickle
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.preprocessing import StandardScaler
import os
import json
import re

# 1. 加载 GloVe 词向量
def load_glove_embeddings(glove_file, embedding_dim):
    """
    加载 GloVe 预训练词向量。
    :param glove_file: GloVe 文件路径
    :param embedding_dim: 词向量维度
    :return: 一个字典 {word: embedding_vector}
    """
    embeddings_index = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector
    print(f"Loaded {len(embeddings_index)} word vectors.")
    return embeddings_index

# 2. 将文本转换为向量
def text_to_vector(text, embeddings_index, embedding_dim, stop_words):
    """
    将文本转化为向量。
    :param text: 输入文本
    :param embeddings_index: 词向量字典
    :param embedding_dim: 词向量维度
    :param stop_words: 停用词列表
    :return: 文本的向量表示
    """
    # 对文本进行预处理：移除标点符号、转换为小写、分词。
    text = re.sub(r'[^\w\s]', '', text)  # 移除标点符号
    text = text.lower()  # 转为小写
    words = text.split()  # 根据空格切分单词
    words = [word for word in words if word not in stop_words]  #过滤停用词
    valid_vectors = [
        embeddings_index[word] for word in words if word in embeddings_index
    ]
    if valid_vectors:
        return np.mean(valid_vectors, axis=0)  # 对所有词向量取均值
    else:
        return np.zeros(embedding_dim)  # 如果没有匹配词，返回全零向量

def transform_texts_to_vectors(texts, embeddings_index, embedding_dim, stop_words):
    """
    将文本列表转换为向量矩阵。
    :param texts: 文本列表
    :param embeddings_index: 词向量字典
    :param embedding_dim: 词向量维度
    :param stop_words: 停用词列表
    :return: 矩阵，每行是一个文本的向量表示
    """
    return np.array([
        text_to_vector(text, embeddings_index, embedding_dim, stop_words) for text in texts
    ])

# 3. 读取数据集（这里假设你已有函数读取数据）
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


# 5. 模型训练和预测
def predict_test_labels():
    # 文件路径
    label_list_file = 'data/label_list.txt'
    train_file = 'data/train.json'
    test_file = 'data/test.txt'
    stopwords_file = 'data/en_stopwords.txt'
    vocab_file = 'data/vocab.pkl'
    model_file = 'data/sgd_model.pkl'
    valid_file = 'data/valid.json'

    # 加载标签列表
    label_list = open(label_list_file).read().splitlines()

    # 加载 GloVe 词向量
    embedding_dim = 300  # GloVe 300d
    glove_file = 'glove.6B.300d.txt'
    embeddings_index = load_glove_embeddings(glove_file, embedding_dim)

    # 加载停用词
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        stop_words = [line.strip().lower() for line in f.readlines()]

    # 加载训练集和测试集
    train_inputs, train_labels = read_dataset(train_file, label_list)
    # test_inputs, _ = read_dataset(test_file, label_list_file)
    test_inputs, labels_true = read_dataset(valid_file, label_list)
    # X_test = ds.transform(test_inputs)
    X_test = transform_texts_to_vectors(test_inputs, embeddings_index, embedding_dim, stop_words)
    Y_true = np.array(labels_true)

    # 将文本转化为词向量
    X_train = transform_texts_to_vectors(train_inputs, embeddings_index, embedding_dim, stop_words)
    # X_test = transform_texts_to_vectors(test_inputs, embeddings_index, embedding_dim)
    Y_train = np.array(train_labels)

    # 特征标准化
    scaler = StandardScaler(with_mean=False)  # 稀疏特征不支持均值中心化
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 初始化 SGDClassifier
    base_model = SGDClassifier(
        loss='hinge',  # 损失函数，可改为 'log'
        penalty='l2',  # 正则化类型
        alpha=0.0001,  # 正则化强度
        max_iter=1000,  # 最大迭代次数
        tol=1e-3,       # 收敛容忍度
        random_state=42
    )

    model = OneVsRestClassifier(base_model)

    # 训练模型
    model.fit(X_train, Y_train)

    # 保存模型
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

    # 预测标签
    Y_pred = model.predict(X_test)
    # 转换稀疏矩阵为密集矩阵
    Y_pred = Y_pred.toarray() if hasattr(Y_pred, 'toarray') else Y_pred
    print(f"Shape of Y_true: {Y_true.shape}")
    print(f"Shape of Y_pred: {Y_pred.shape}")

    # 评估性能（如果有真实标签）
    from sklearn.metrics import accuracy_score, f1_score, hamming_loss, recall_score, precision_score
    # 假设测试集真实标签为 Y_true
    accuracy = accuracy_score(Y_true, Y_pred)
    f1 = f1_score(Y_true, Y_pred, average='macro')
    hamming = hamming_loss(Y_true, Y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    recall = recall_score(Y_true, Y_pred, average='macro')  # 宏平均召回率
    precision = precision_score(Y_true, Y_pred, average='macro')  # 宏平均精确率
    print(f"precision: {precision:.4f}")
    print(f"recall: {recall:.4f}")
    print(f"F1 Score (Macro): {f1:.4f}")
    print(f"Hamming Loss: {hamming:.4f}")



    # 转换预测结果为标签名
    index_to_label = {i: label for i, label in enumerate(label_list)}
    predictions = []
    for pred in Y_pred:
        pred_labels = [index_to_label[i] for i, val in enumerate(pred) if val == 1]
        predictions.append(pred_labels)

    # 保存预测结果
    # with open('predictions.txt', 'w', encoding='utf-8') as f:
    #     for idx, pred_labels in enumerate(predictions):
    #         f.write(f" {', '.join(pred_labels)}\n")
    # print("预测完成，结果已保存到 predictions.txt 文件。")
    with open('predictions.txt', 'w', encoding='utf-8') as f:
        for pred_labels in predictions:
            # 如果预测标签为空，写入 "No_Mentioned"
            if not pred_labels:
                f.write("No_Mentioned\n")
            else:
                f.write(f"{','.join(pred_labels)}\n")
    print("预测完成，结果已保存到 predictions.txt 文件。")

# 运行预测
if __name__ == '__main__':
    predict_test_labels()
