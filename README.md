# NLP
基于 GloVe 词向量和 TF-IDF 特征的多标签文本分类模型对比
一.数据集
数据集中包含10150个训练集，2817个验证集，2540个测试集。隐私政策文本分类任务是多分类问题，有31个类别标签，在label_list里面可以看到。训练集和验证集是两个json文件，一行是一个样本
二.模型训练
本实验可以基于TF-IDF特征采用三种模型（多项式核 SVM、LinearSVC  和 SGDClassifier），在data4ml.py中进行数据处理，在ml_predict.py分别测试自己想使用的模型模块。有以下几个注意事项：
1.使用一种模型时要将其他模型注释掉，同时删除上一个模型训练生成的文件multi_lable_model.pkl，其中存放了已经训练好的模型。
2.对于LinearSVC模型，会生成最优模型，故提取模型时与其他两种不同，best_model = model.best_estimator_
本实验还可以基于GloVe词向量进行SGDClassifier多分类任务，在DL_predict.py文件中，但是效果不佳
三.测试
可以在ml_predict.py或者DL_predict.p中的标签预测模块，选择自己想预测的数据集。预测测试集test.txt时，与预测验证集时，对测试数据的加载是不同的，要使用对应模块，并把另一个注释掉。
