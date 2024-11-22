from data4ml import read_dataset, read_label_list, MyDataset
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, hamming_loss, f1_score
import pickle
import numpy as np
import os


def predict_test_labels():
    # 文件路径
    label_list_file = 'data/label_list.txt'
    train_file = 'data/train.json'
    valid_file = 'data/valid.json'
    test_file = 'data/test.txt'
    stopwords_file = 'data/en_stopwords.txt'
    vocab_file = 'data/vocab.pkl'
    model_file = 'data/multi_label_model.pkl'

    # 加载标签列表
    label_list = read_label_list(label_list_file)

    # 初始化数据处理类
    ds = MyDataset()

    # 检查是否有缓存,如果模型和词典缓存文件存在，直接加载缓存，跳过训练过程。
    if all(map(os.path.exists, [vocab_file, model_file])):
        with open(vocab_file, 'rb') as f:
            ds.vocab = pickle.load(f)
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
    else:
        # 加载训练数据和验证数据
        train_inputs, train_labels = read_dataset(train_file, label_list)
        valid_inputs, valid_labels = read_dataset(valid_file, label_list)

        # 构建词典，使用 停用词表过滤掉停用词
        ds.set_stopword(stopwords_file)
        ds.build_tfidf_vocab(train_inputs, max_size=5000)

        # 转换训练集和验证集为TF-IDF 特征向量。
        X_train = np.array(ds.transform(train_inputs))
        Y_train = np.array(train_labels)
        X_valid = np.array(ds.transform(valid_inputs))
        Y_valid = np.array(valid_labels)

        # 使用验证集进行超参数调优
        # base_model = LinearSVC(C=1, max_iter=5000)
        # param_grid = {'estimator__C': [ 0.1, 1, 10]}  # 超参数网格
        # model = GridSearchCV(
        #     OneVsRestClassifier(base_model),
        #     param_grid,
        #     scoring='f1_macro',  # 使用 F1-score 作为评分指标
        #     cv=3  # 在验证集上进行交叉验证
        # )

        # 初始化 SGDClassifier
        # base_model = SGDClassifier(
        #     loss='hinge',  # 损失函数，可改为 'log'
        #     penalty='l2',  # 正则化类型
        #     alpha=0.0001,  # 正则化强度
        #     max_iter=1000,  # 最大迭代次数
        #     tol=1e-3,  # 收敛容忍度
        #     random_state=42
        # )
        # model = OneVsRestClassifier(base_model)

        # 使用多项式核的SVM
        base_model = SVC(kernel='poly', degree=3, coef0=1, C=1, random_state=42)  # 多项式核SVM，degree是多项式的度
        model = OneVsRestClassifier(base_model)

        # 训练模型
        model.fit(X_train, Y_train)

        # 从 GridSearchCV 中提取最佳模型
        # best_model = model.best_estimator_
        # # 输出最佳超参数和验证集上的性能
        # print(f"Best parameters: {model.best_params_}")
        # print(f"Validation F1-score: {model.best_score_:.4f}")

        # 保存模型和词典
        with open(vocab_file, 'wb') as f:
            pickle.dump(ds.vocab, f)
        with open(model_file, 'wb') as f:
            # pickle.dump(model.best_estimator_, f)
            pickle.dump(model, f)


    # 加载测试数据
    # test_inputs, _ = read_dataset(test_file, label_list)
    # X_test = ds.transform(test_inputs)
    # 加载并处理测试集
    # test_file_path = r'data\test.txt'
    # with open(test_file_path, 'r', encoding='utf-8') as f:
    #     test_inputs = [line.strip() for line in f]
    # X_test = ds.transform(test_inputs)
    # 加载测试数据,使用验证集进行测试
    test_inputs, labels_true = read_dataset(valid_file, label_list)
    X_test = ds.transform(test_inputs)
    Y_true = np.array(labels_true)

    # 预测标签 使用训练好的模型 model 对 X_test 进行预测，返回一个二值矩阵 Y_pred。
    Y_pred = model.predict(X_test)
    # print(f"Shape of Y_true: {Y_true.shape}")
    # print(f"Shape of Y_pred: {Y_pred.shape}")
    # 使用最佳模型对测试数据进行预测
    # Y_pred = best_model.predict(X_test)

    # 评估性能（如果有真实标签）
    from sklearn.metrics import accuracy_score, f1_score, hamming_loss, recall_score, precision_score
    # 假设测试集真实标签为 Y_true
    accuracy = accuracy_score(Y_true, Y_pred)
    f1 = f1_score(Y_true, Y_pred, average='macro')
    hamming = hamming_loss(Y_true, Y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    recall = recall_score(Y_true, Y_pred, average='macro', zero_division=0)  # 宏平均召回率
    precision = precision_score(Y_true, Y_pred, average='macro', zero_division=0)  # 宏平均精确率
    print(f"precision(Macro): {precision:.4f}")
    print(f"recall(Macro): {recall:.4f}")
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
    #         f.write(f"{','.join(pred_labels)}\n")
    # print("预测完成，结果已保存到 predictions.txt 文件。")
    # 保存预测结果，避免空行
    with open('predictions.txt', 'w', encoding='utf-8') as f:
        for pred_labels in predictions:
            # 如果预测标签为空，写入 "No_Mentioned"
            if not pred_labels:
                f.write("No_Mentioned\n")
            else:
                f.write(f"{','.join(pred_labels)}\n")
    print("预测完成，结果已保存到 predictions.txt 文件。")


if __name__ == '__main__':
    predict_test_labels()
