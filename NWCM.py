import numpy as np
import random as rd
import pandas as pd
import copy
import matplotlib.pyplot as plt
from minepy import MINE
from more_itertools import distinct_combinations
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler


class CFS():
    '''
    *Correlation-based Feature Selection*
    dataset: 样本集
    label: 标签
    '''
    def __init__(self):
        self.dataset = None
        self.label = None
        self.leastFeatInd = None            # 得分最低的特征
        self.scores = None                   # 拟合后的得分，list类型

    def fit(self, dataset, label):
        '''拟合数据集'''
        self.scores = []                      #各特征得分
        self.dataset = dataset
        self.label = label
        X = dataset
        Y = label
        assert len(X) == len(Y)
        feature_size = X.shape[1]
        correlations = np.corrcoef(X, Y, rowvar = False)     #pearson相关系数
        
        k = len(X[0])
        if feature_size == 1:
            self.scores.append(correlations[0][-1])
            self.leastFeatInd = 0
        else:
            for i in range(feature_size):
                #求出各特征子集的得分(即按某特征划分后的特征子集)
                temp_relavent_ind = [j for j in range(feature_size)]                    # 包含选中特征的列号的列表
                del temp_relavent_ind[i]
                if feature_size == 2: row_ind, col_ind = temp_relavent_ind[0], temp_relavent_ind[0]
                else: 
                    row_ind, col_ind = zip(*distinct_combinations(temp_relavent_ind, 2))    # 选中特征的全排列
                ff_mean = correlations[row_ind, col_ind].mean()                         # 特征-特征的平均相关性
                fc_mean = correlations[temp_relavent_ind, -1].mean()                    # 特征-类的平均相关性
                merits = (k * fc_mean) / np.sqrt(k + k * (k - 1) * ff_mean)
                self.scores.append(merits)
            self.leastFeatInd = np.argmin(np.array(self.scores))

class Chi2():
    '''
    *卡方分布特征选择*
    '''
    def __init__(self):
        self.dataset = None
        self.label = None
        self.leastFeatInd = None            # 得分最低的特征
        self.scores = None                   # 拟合后的得分，list类型

    def fit(self, dataset, label):
        '''拟合数据集'''
        self.dataset = dataset
        self.label = label
        chi = SelectKBest(chi2, k = 'all')
        chi.fit(dataset, label)
        self.leastFeatInd = np.argmin(chi.scores_)
        self.scores = list(chi.scores_)

class ReliefF():
    '''
    *ReliefF*
    '''
    def __init__(self):
        self.dataset = None
        self.label = None
        self.leastFeatInd = None            # 得分最低的特征的索引
        self.scores = None                   # 拟合后的得分，list类型

    def fit(self, dataset, label, m = 30, k = 20):
        '''
        dataset: 数据集
        label: 对应每个样本的标签
        m: 样本抽样次数
        k: 最近邻样本个数
        '''
        def diff(A, E1, E2):
            '''求出两向量间的差异值'''
            d = np.linalg.norm(E1 - E2, ord = 1) / (np.max(dataset[:, A]) - min(dataset[:, A]))
            return d
        self.dataset = dataset
        self.label = label
        data_size = len(dataset)
        feature_size = len(dataset[0])
        unique_label = set(label)
        W = [0 for i in range(feature_size)]                #用于返回的各特征权重
        split_dataset = {l:[] for l in unique_label}        #按每个特征划分数据集,字典存储样本
        for i in range(data_size):
            split_dataset[label[i]].append(dataset[i])

        for i in range(m):                                      # m次抽样
            for feature_ind in range(feature_size):             # 更新每个特征的权重
                R_ind = rd.randint(0, data_size - 1)            # 随机抽取一个样本的行号
                R_label = label[R_ind]                          # R的标签
                Hits = np.array(split_dataset[R_label])               # 同类样本
                Misses = {key: np.array(split_dataset[key]) for key in split_dataset.keys() if key != R_label}

                # 一阶范数寻找R的k个同类最近邻样本，和每个与R不同类的k个样本(共n*k个)
                Hits = Hits[np.linalg.norm(Hits - dataset[R_ind], ord = 1, axis = 1).argsort()] 
                near_Hits = Hits[:k]
                near_Misses = {key: value[np.linalg.norm(value - dataset[R_ind], ord = 1, axis = 1).argsort()][:k] for key, value in Misses.items()}
                sum_diffRH = 0
                sum_diffRM = 0
                for j in range(k):
                    sum_diffRH += diff(feature_ind, dataset[R_ind], near_Hits[j])
                for cla in list(unique_label):
                    if cla == R_label: continue
                    for j in range(k):
                        sum_diffRM += (len(split_dataset[cla]) / (data_size - len(split_dataset[R_label]))) * diff(feature_ind, dataset[R_ind], near_Misses[cla][j])
                        
                W[feature_ind] += (sum_diffRM -  sum_diffRH) / m * k        # 权重更新公式

        self.scores = W
        self.leastFeatInd = np.argmin(np.array(W))

class MIFS():
    '''
    *Mutual Information Feature Selection*
    '''
    def __init__(self):
        self.dataset = None
        self.label = None
        self.leastFeatInd = None            # 得分最低的特征
        self.scores = None                   # 拟合后的得分，list类型

    def fit(self, dataset, label, beta = 0.8):
        '''
        dataset: 只包含特征的样本
        label: 对应样本的标签
        beta: 惩罚系数(一般取0.5~1)
        '''
        feature_size = len(dataset[0])
        X = np.hstack([dataset, label.reshape((dataset.shape[0], 1))])      # 合并数据与标签
        MI_mat = np.zeros((feature_size + 1, feature_size + 1))             # 互信息矩阵(不计算自己与自己)
        for i in range(feature_size):
            for j in range(i + 1, feature_size + 1):
                mine = MINE(alpha = 0.6, c = 15)
                mine.compute_score(X[:, i], X[:, j])
                MI_mat[i, j] = mine.mic()
        J = [0 for i in range(feature_size)]
        features = [i for i in range(feature_size)]
        for feat in range(feature_size):                                   # 挑选出n - 1个特征，剩下一个作为划分特征
            features.pop(0)
            J[feat] = MI_mat[feat, -1] - beta * (MI_mat[feat, features].sum())
            features.append(feat)
        self.scores = J
        self.leastFeatInd = np.argmin(np.array(J))


class NWCM():
    '''
    *A Novel Weighted Combination Method for Feature Selection using Fuzzy Sets*
    '''
    def __init__(self):
        self.L = 0                          # boostrap划分子集数
        self.dataset = None                 # 数据集
        self.label = None                   # 样本标签
        self.leastFeatInd = None            # 最不重要特征的列号
        self.FS_num = 4                     # 特征选择方法数
        self.c = None                       # 最终特征得分

    def generateFS(self, dataset, label, subset_size, m, k, beta):
        '''
        *生成模糊集*
        返回一个列表和标准化后的特征得分，第一维是特征，第二维是特征选择方法，第三维每个子集l对应U集
        dataset: 数据集
        label: 样本标签
        subset_size: int型，boostrap每个子集的大小
        m: ReliefF样本抽样次数
        k: ReliefF样本最近邻样本个数
        beta: MIFS惩罚系数
        '''
        subset = []
        sublabel = []
        feat_size = len(dataset[0])                     # 特征数
        for i in range(self.L):
            row_sequence = np.random.choice(dataset.shape[0], subset_size, replace = True, p = None)  # 抽取size大小的子集(有放回重复抽样)
            subset.append(dataset[row_sequence, :])
            sublabel.append(label[row_sequence])
        Scores = [[] for i in range(self.FS_num)]                  # 特征得分，第一维是特征选择方法，第二维是样本子集，第三维是各特征得分
        # 定义4种特征选择方法的分类器
        cfs = CFS()
        chi2_fs = Chi2()
        reliefF = ReliefF()
        mifs = MIFS()
        # 计算各个FS的特征得分
        for ind in range(self.L):
            cfs.fit(subset[ind], sublabel[ind])
            chi2_fs.fit(subset[ind], sublabel[ind])
            reliefF.fit(subset[ind], sublabel[ind], m, k)
            mifs.fit(subset[ind], sublabel[ind], beta)
            Scores[0].append(cfs.scores)
            Scores[1].append(chi2_fs.scores)
            Scores[2].append(reliefF.scores)
            Scores[3].append(mifs.scores)
        
        # 特征得分归一化处理
        Scores = np.array(Scores)
        # c = np.mean(Score, axis = 1)            # 求出各方法在所有子集下的平均特征得分

        for func_ind in range(Scores.shape[0]):                      # 对每种特征选择方法
            maxcols = Scores[func_ind].max(axis = 0)
            mincols = Scores[func_ind].min(axis = 0)
            for col in range(feat_size):                                    # 对每列进行max-min归一化处理
                Scores[func_ind][:, col] = (Scores[func_ind][:, col] - mincols[col]) / (maxcols[col] - mincols[col])
        # 归整化小数后两位
        Scores = np.round(Scores, decimals = 2)
        # 构建模糊集
        FuzzySet = np.zeros((self.FS_num, feat_size, 101), float)     # 第一维是特征选择方法，第二维是特征，第三维每个子集l对应U集
        for col in range(feat_size):
            for func_ind in range(self.FS_num):                               # 统计各段频数
                for row in range(self.L):
                    gap = int(Scores[func_ind][row][col] / 0.01)
                    FuzzySet[func_ind][col][gap] += 1
        FuzzySet = FuzzySet / self.L
        return FuzzySet, Scores

    def WeightedCombine(self, FuzzySet, Scores, methods):
        '''
        *各模糊集的带权结合*
        FuzzySet: 模糊集
        Score: 标准化后的特征得分
        methods: 选择的权重选择方法(list)
        返回带权联合模糊集，第一维是特征，第二维是每个子集l对应的U集
        '''
        # Equal Weights
        def EW(FuzzySet, Scores):
            weight_EW = 1 / self.FS_num
            EW = np.sum(weight_EW * FuzzySet, axis = 0)
            return EW

        # Reciprocal Standard Deviation Weights
        def RW(FuzzySet, Scores):
            SD = np.std(Scores, axis = 1)                # 标准差矩阵，第一维是特征选择方法，第二维是特征
            recip = 1 / SD                              # 标准差倒数
            weight_RW = recip / np.sum(recip, axis = 0)
            weight_RW = np.repeat(weight_RW, 101).reshape(FuzzySet.shape)
            RW = np.sum(weight_RW * FuzzySet, axis = 0)
            return RW

        # One Minus Standard Deviation Weights
        def OW(FuzzySet, Scores):
            SD = np.std(Scores, axis = 1)                # 标准差矩阵，第一维是特征选择方法，第二维是特征
            weight_OW = (1 - SD) / np.sum(1 - SD, axis = 0)
            weight_OW = np.repeat(weight_OW, 101).reshape(FuzzySet.shape)
            OW = np.sum(weight_OW * FuzzySet, axis = 0)
            return OW

        # Matrix Similarity Weights
        def MW(FuzzySet, Scores):
            feat_size = FuzzySet.shape[1]
            BinaryMatrix = np.zeros((self.FS_num, feat_size, self.L, 101))         # 二进制矩阵，第一维是特征选择方法，第二维是特征，第三维是p/L，第四维是U(0,0.01..1)
            for func_ind in range(self.FS_num):
                for feat_ind in range(feat_size):
                    for p in range(self.L):
                        for q in range(101):
                            if (p + 1) / self.L <= FuzzySet[func_ind][feat_ind][q]: BinaryMatrix[func_ind][feat_ind][p][q] = 1
            comMatrix = np.sum(BinaryMatrix, axis = 0)              # 按特征方法求和，即第一维求和
            comMatrix = np.stack([comMatrix for i in range(self.FS_num)], axis = 0)    # 合并数组
            simil = BinaryMatrix * comMatrix
            v = np.linalg.norm(simil, ord = 1, axis = (2,3)) / np.linalg.norm(comMatrix, ord = 1, axis = (2,3))     # 第一维是特征方法，第二维是特征
            weight_MW = v / np.sum(v, axis = 0)
            weight_MW = np.repeat(weight_MW, 101).reshape(FuzzySet.shape)
            MW = np.sum(weight_MW * FuzzySet, axis = 0)
            return MW
        meth = {'EW': EW, 'RW': RW, 'OW': OW, 'MW': MW}
        return meth[methods](FuzzySet, Scores)


    def defuzzificate(self, comFuzzySet):
        '''
        *去离散化，返回得分最低的特征序号*
        comFuzzySet: 联合模糊集，第一维是特征，第二维是U集
        '''
        row, col = comFuzzySet.shape
        self.c = np.zeros(row)
        for i in range(row):
            t = np.sum(comFuzzySet[i])
            for j in range(col):
                self.c[i] = j * 0.01 * comFuzzySet[i][j] / t
        return np.argmin(self.c)

    def fit(self, dataset, label, L, subset_size, m, k, beta = 0.8, w_meth = 'EW'):
        '''
        *适应训练集*
        L: 子集数量
        subset_size: 每个子集的规模
        m: ReliefF抽样次数
        k: ReliefF最近邻样本数
        beth: MIFS惩罚系数
        w_meth: 权重分配方式(目前有EW、RW、OW、MW)
        '''
        self.dataset = dataset
        self.label = label
        self.L = L

        FuzzySet, Scores = self.generateFS(dataset, label, subset_size, m, k, beta)
        comFuzzySet = self.WeightedCombine(FuzzySet, Scores, w_meth)   # 不同权重分配方法下的联合模糊集
        self.leastFeatInd = self.defuzzificate(comFuzzySet)

def LineChartPlotter(x_data, y_data, x_label, y_label, title, axis_range, legends, linestyles, save_path):
    '''
    *绘制折线图*
    x_data: 横坐标(list)
    y_data: 纵坐标(list)
    x_label: 横轴标签
    y_label: 纵轴标签
    title: 折线图标题
    axis_range: 横纵坐标显示范围[xs,xe, ys, ye]
    legend: 折线标识
    linestyles: 指定每条折线的样式
    save_path: 折线图保存路径
    '''
    assert(len(x_data) == len(y_data))
    assert(len(x_data) == len(linestyles))
    line_num = len(x_data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.axis(axis_range)
    for i in range(line_num):
        plt.plot(x_data[i], y_data[i], linestyle = linestyles[i])
    plt.legend(legends)
    plt.savefig(save_path)
    plt.show()
    

def APC(K, X, Y, FSMethods, W_Methods, title, legend, path):
    '''
    *计算APC (average Pearson's Correlation)，并绘制相应图像*
    K: k折检验次数
    X: 预处理后的数据集
    Y: 样本标签
    FSMethods: 包含所有FS类的list
    W_Methods: 包含所有NWCM权重分配方法的'str'
    title: 图片标题
    legend: 折线标识
    path: 图片存储路径
    '''
    NWCM_cf = NWCM()
    single_APC = []         # 单独FS下的APC，第一维是FS方法，第二维是各子集规模下的APC
    NWCM_APC = []           # NWCM下的APC，第一维是权重方法，第二维是各子集规模下的APC
    for meth in FSMethods:
        EachSizeAPC = []                                                # 每个数据集大小下的APC
        for dataset_size in list(np.arange(0.9, 0.1, -0.1)):               # 调整测试集大小
            kc = []                                                     # k折特征得分
            for k in range(K):
                train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = dataset_size)
                meth.fit(train_X, train_Y)
                kc.append(meth.scores)
            kc = np.array(kc)
            corr_Scores = np.corrcoef(kc, rowvar = False)                   # 相关系数矩阵
            row_ind, col_ind = zip(*distinct_combinations([i for i in range(X.shape[1])], 2))    # 选中特征的全排列
            APC = np.sum(corr_Scores[row_ind, col_ind]) / (K * (K - 1) / 2)
            EachSizeAPC.append(APC)
        single_APC.append(EachSizeAPC)
        print(meth.__class__.__name__)
        print(EachSizeAPC)

    for meth in W_Methods:
        EachSizeAPC = []                                                # 每个数据集大小下的APC
        for dataset_size in list(np.arange(0.9, 0.1, -0.1)):               # 调整测试集大小
            print('dataset_size: ', dataset_size)
            kc = []                                                     # k折特征得分
            for k in range(K):                                          # 5折检验
                print("正在进行第", k + 1, "折..")
                train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = dataset_size)
                NWCM_cf.fit(train_X, train_Y, L = 100, subset_size = int(0.3 * X.shape[0]), m = 30, k = 20, beta = 0.8, w_meth = meth)
                kc.append(NWCM_cf.c)
            kc = np.array(kc)
            corr_Scores = np.corrcoef(kc, rowvar = False)                   # 相关系数矩阵
            row_ind, col_ind = zip(*distinct_combinations([i for i in range(X.shape[1])], 2))    # 选中特征的全排列
            APC = np.sum(corr_Scores[row_ind, col_ind]) / (K * (K - 1) / 2)
            EachSizeAPC.append(APC)
            print(APC)
        NWCM_APC.append(EachSizeAPC)
        print(meth.__class__.__name__)
        print(EachSizeAPC)
    #绘制折线图
    reduce_size = [list(np.arange(0.9, 0.1, -0.1)) for _ in range(8)]
    LineChartPlotter(reduce_size, single_APC + NWCM_APC, x_label = 'The proportion of the data for training', y_label = "Pearson's Correlation of the Feature Scores", \
                     title = title, axis_range = [0, pre_X.shape[1] - 1, 0.5, 0.9], legends = legend, \
                     linestyles = ['-.', '-.','-.', '-.', '-', '-','-', '-'], save_path = path)

def ACC(X, Y, FSMethods, W_Methods, title, legend, path):
    '''
    *准确率与移除特征数关系*
    X: 预处理后的数据集
    Y: 样本标签
    FSMethods: 包含所有FS类的list
    W_Methods: 包含所有NWCM权重分配方法的'str'
    title: 图片标题
    legend: 折现标识
    path: 图片存储路径
    '''
    NWCM_cf = NWCM()
    NB = GaussianNB()
    rf = RandomForestClassifier(n_estimators = 20, max_features = None)

    # 单独使用4种FS进行分类
    single_FS_acc = []                          # 单独FS准确率
    for meth in FSMethods:
        X = copy.deepcopy(pre_X)
        y_acc = []                                # 分类准确率
        for i in range(X.shape[1] - 1):
            train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = 0.35, random_state = 0)
            
            # rf.fit(train_X, train_Y)
            # pr_RF = rf.predict(test_X)              # RF预测结果
            # y_acc.append(accuracy_score(test_Y, pr_RF))

            NB.fit(train_X, train_Y)
            pr_NB = NB.predict(test_X)            # NB预测结果
            y_acc.append(accuracy_score(test_Y, pr_NB))
            
            meth.fit(train_X, train_Y)
            remove_featInd = meth.leastFeatInd
            X = np.delete(X, remove_featInd, axis = 1)
        single_FS_acc.append(y_acc)
        print(meth.__class__.__name__)
        print(y_acc)
    print(single_FS_acc)

    # 使用NWCM，配下，搭配NB进行分类
    NWCM_acc = []                                   # NWCM准确率
    for meth in W_Methods:
        print(meth)
        X = copy.deepcopy(pre_X)
        y_acc = []                              # 折线图meth权重下的精确度
        for i in range(X.shape[1] - 1):
            print('第', i + 1, '组: ')
            train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = 0.35, random_state = 0)
            
            NB.fit(train_X, train_Y)
            pr_NB = NB.predict(test_X)      # NB预测结果
            acc_score = accuracy_score(test_Y, pr_NB)
            y_acc.append(acc_score)
            print('NB: ', acc_score)
            print(classification_report(test_Y, pr_NB))
            
            # rf.fit(train_X, train_Y)
            # pr_RF = rf.predict(test_X)              # RF预测结果
            # acc_score = accuracy_score(test_Y, pr_RF)
            # y_acc.append(acc_score)
            # print('RF: ', acc_score)
            # print(classification_report(test_Y, pr_RF))

            # 使用NWCM选出最优划分特征(即得分最低的特征)
            NWCM_cf.fit(X, Y, L = 100, subset_size = int(0.3 * X.shape[0]), m = 30, k = 20, beta = 0.8, w_meth = meth)
            remove_featInd = NWCM_cf.leastFeatInd
            X = np.delete(X, remove_featInd, axis = 1)
        NWCM_acc.append(y_acc)
    print(NWCM_acc)
    # 绘制折线图
    remove_features = [[i for i in range(pre_X.shape[1] - 1)] for _ in range(8)]        # 折线图移除的特征数(8条线)
    LineChartPlotter(remove_features, single_FS_acc + NWCM_acc, x_label = 'The number of the removed features', y_label = 'Classification Accuracy', \
                     title = title, axis_range = [0, pre_X.shape[1] - 1, 0.5, 0.8], legends = legend, \
                     linestyles = ['-.', '-.','-.', '-.', '-', '-','-', '-'], save_path = path)

if __name__ == '__main__':
    # 糖尿病
    acc_NB_title = 'Comparison using NB in PIMA'
    acc_NB_path = './Comparison using NB in PIMA.jpg'
    acc_RF_title = 'Comparison using RF in PIMA'
    acc_RF_path = './Comparison using RF in PIMA.jpg'
    apc_title = 'Comparison on Reduced Sizes of PIMA Dataset'
    apc_path = './Comparison on Reduced Sizes of PIMA Dataset'
    PI_dataset = pd.read_csv("./data/PimaIndiansdiabetes.csv")
    dataset = PI_dataset.to_numpy()
    X = dataset[:, :-1]
    Y = dataset[:, -1].astype(int)
    
    '''
    # 乳腺癌
    # acc_NB_title = 'Comparison using NB in BCW'
    # acc_NB_path = './data/Comparison using NB in BCW.jpg'
    # title = 'Comparison using RF in BCW'
    # path = './Comparison using RF in BCW.jpg'
    BCW_dataset = pd.read_csv("./data/breast-cancer-wisconsin.csv", header = None)
    BCW_dataset = BCW_dataset.drop(columns = 0)
    BCW_mean = np.array(BCW_dataset.mean()).astype('int')
    dataset = BCW_dataset.to_numpy()
    # 使用平均值替换空缺值'?'
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[1] - 1):
            if isinstance(dataset[i][j], str): dataset[i][j] = BCW_mean[j]
    X = dataset[:, :-1].astype(int)
    Y = dataset[:, -1].astype(int)
    '''
    
    # 帕金森症
    # acc_NB_title = 'Comparison using NB in Parkinsons'
    # acc_NB_path = './Comparison using NB in Parkinsons.jpg'
    # acc_RF_title = 'Comparison using RF in Parkinsons'
    # acc_RF_path = './Comparison using RF in Parkinsons.jpg'
    # PKS_dataset = pd.read_csv("./data/parkinsons.csv")
    # PKS_dataset = PKS_dataset.drop(columns = ['name'])
    # dataset = PKS_dataset.to_numpy()
    # X = dataset[:, :-1].astype(float)
    # X = np.abs(X)
    # Y = dataset[:, -1].astype(int)
    
    # 最大最小标准化
    scaler = MinMaxScaler()
    pre_X = scaler.fit_transform(X)
    # pre_X = X           # 处理后的数据集

    cfs_fs = CFS()
    chi2_fs = Chi2()
    reliefF_fs = ReliefF()
    mifs_fs = MIFS()
    FSMethods = [cfs_fs, chi2_fs, reliefF_fs, mifs_fs]        # 4种FS方法
    W_Methods = ['EW', 'RW', 'OW', 'MW']                      # 权重分配方法
    legend = ['EW', 'RW', 'OW', 'MW', 'cfs', 'chi2', 'ReliefF', 'MIFS']   # 设置线条注解
    ACC(pre_X, Y, FSMethods, W_Methods, title = acc_NB_title, legend = legend, path = acc_NB_path)
    # ACC(pre_X, Y, FSMethods, W_Methods, title = acc_RF_title, legend = legend, path = acc_RF_path)
    #APC(5, pre_X, Y, FSMethods, W_Methods, title = apc_title, legend = legend, path = apc_path)
    
    
    # LineChartPlotter([[1,2,3,4,5] for _ in range(2)], [[2,3,4,5,6],[3,4,5,6,7]], x_label = 'xlabel', \
    #                   y_label = 'ylabel', title = 'abc', axis_range = [1,5,2,8], legends = ['A', 'B'], linestyles = ['-', '-.'], \
    #                   save_path = './testPhoto.jpg')
