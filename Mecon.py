import matplotlib #  导入matplotlib库，一个常用于绘制图表的Python库。

matplotlib.use('agg') # 设置matplotlib的后端为'agg'。这是一个用于非交互式环境（如脚本或服务器）的后端，不需要GUI。
import numpy as np
import time
import os
import torch.utils.data
import torch.nn.functional as F # 导入PyTorch中的函数式接口，简称为F。
import torch.optim as optim # 导入PyTorch中的优化算法模块。
import torch.optim.lr_scheduler as lr_scheduler # 导入PyTorch中的学习率调整策略模块。
from torch.utils.data import DataLoader # 从PyTorch中导入DataLoader类，用于批量加载数据。
from os.path import join as pjoin # 导入join函数，并命名为pjoin。用于路径拼接。
from parser import parameter_parser # 导入一个自定义的parser模块，用于解析参数。
from load_data import split_ids, GraphData, collate_batch # 导入自定义的数据加载模块。
from models.gcn_modify import GCN_MODIFY # 导入自定义的修改版图卷积网络(GCN)模型。
from models.gcn_origin import GCN_ORIGIN # 导入原始的图卷积网络(GCN)模型。
from models.gat import GAT # 导入图注意力网络(GAT)模型。
from sklearn import metrics # 导入scikit-learn的度量模块，用于评估模型性能。

print('using torch', torch.__version__)  # 打印当前使用的PyTorch版本
args = parameter_parser()  # 使用参数解析器解析命令行参数
args.filters = list(map(int, args.filters.split(',')))  # 将filters参数转换为整数列表
args.lr_decay_steps = list(map(int, args.lr_decay_steps.split(',')))  # 将lr_decay_steps参数转换为整数列表
for arg in vars(args):
    print(arg, getattr(args, arg))  # 打印所有解析的参数

n_folds = args.folds  # 从参数中获取交叉验证的折数
torch.backends.cudnn.deterministic = True  # 设置cudnn为确定性计算，保证实验可重复
torch.backends.cudnn.benchmark = True  # 启用cudnn性能优化
torch.manual_seed(args.seed)  # 设置PyTorch的全局随机种子
torch.cuda.manual_seed(args.seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(args.seed)  # 为所有GPU设置随机种子
rnd_state = np.random.RandomState(args.seed)  # 创建NumPy的随机状态生成器

print('Loading training_data...')  # 打印加载训练数据的提示信息

# 这个类负责读取和处理图形数据，加载和预处理数据，将数据转换为神经网络可处理的格式。
# 它处理节点特征、邻接矩阵和标签，并支持连续节点属性。
class DataReader():
    """
    Class to read the txt files containing all training_data of the dataset
    DataReader类用于读取和处理图形数据集
    """

    def __init__(self, data_dir, rnd_state=None, use_cont_node_attr=False, folds=n_folds):
        """
        初始化函数
        :param data_dir: 数据目录
        :param rnd_state: 随机状态生成器
        :param use_cont_node_attr: 是否使用连续节点属性
        :param folds: 数据划分的折数
        """
        self.data_dir = data_dir  # 设置数据目录
        # 如果未提供随机状态，则创建一个新的
        self.rnd_state = np.random.RandomState() if rnd_state is None else rnd_state
        self.use_cont_node_attr = use_cont_node_attr  # 设置是否使用连续节点属性
        files = os.listdir(self.data_dir)  # 列出数据目录下的所有文件
        data = {}  # 创建一个空字典用于存储数据

        # 读取和处理图指示文件
        nodes, graphs, unique_id = self.read_graph_nodes_relations(
            list(filter(lambda f: f.find('graph_indicator') >= 0, files))[0])

        # 读取节点特征
        data['features'] = self.read_node_features(
            list(filter(lambda f: f.find('node_labels') >= 0, files))[0],
            nodes, graphs, fn = lambda s: int(s.strip()))

        # 读取图的邻接列表
        data['adj_list'] = self.read_graph_adj(
            list(filter(lambda f: f.find('_A') >= 0, files))[0], nodes, graphs)

        # print(data['adj_list'] )
        # 读取图的目标标签
        data['targets'] = np.array(
            self.parse_txt_file(
                list(filter(lambda f: f.find('graph_labels') >= 0, files))[0],
                line_parse_fn = lambda s: int(float(s.strip()))))

        data['ids'] = unique_id  # 保存唯一标识符

        # 如果使用连续节点属性，则读取相应的特征
        if self.use_cont_node_attr:
            data['attr'] = self.read_node_features(
                list(filter(lambda f: f.find('node_attributes') >= 0, files))[0],
                nodes, graphs,
                fn = lambda s: np.array(list(map(float, s.strip().split(',')))))

        # 处理和转换特征数据
        features, n_edges, degrees = [], [], []
        for sample_id, adj in enumerate(data['adj_list']):
            # print()
            # 计算当前图的节点数
            N = len(adj)
            # number of nodes
            if data['features'] is not None:
                # 确保特征的长度与节点数相匹配
                assert N == len(data['features'][sample_id]), (N, len(data['features'][sample_id]))
            n = np.sum(adj)  # 计算邻接矩阵中边的总数
            n_edges.append(int(n / 2))  # 由于是无向图，边的数量是总和的一半
            # print(adj)
            # print(adj.T)
            if not np.allclose(adj, adj.T):
                # 检查邻接矩阵是否为对称矩阵，无向图的邻接矩阵应该是对称的
                print(sample_id, 'not symmetric')
                # 检查不对称的元素
                for i in range(adj.shape[0]):
                    for j in range(adj.shape[1]):
                        if adj[i, j] != adj[j, i]:
                            print(f"Element ({i}, {j}) is not symmetric. Values: adj[{i}, {j}] = {adj[i, j]}, adj[{j}, {i}] = {adj[j, i]}")

            # exit();
            degrees.extend(list(np.sum(adj, 1)))  # 计算每个节点的度（即每行的和）
            features.append(np.array(data['features'][sample_id]))  # 将特征添加到特征列表中

        # Create features over graphs as one-hot vectors for each node
        # 将特征转换为one-hot编码
        features_all = np.concatenate(features) # 将所有特征合并为一个数组
        features_min = features_all.min()  # 找到特征值的最小值
        num_features = int(features_all.max() - features_min + 1)  # 计算特征的可能值数量

        features_onehot = [] # 初始化one-hot特征列表
        for i, x in enumerate(features):
            # 遍历每个图的特征
            feature_onehot = np.zeros((len(x), num_features))   # 创建一个零矩阵作为one-hot编码基础
            for node, value in enumerate(x):
                # 为每个节点创建one-hot编码
                feature_onehot[node, value - features_min] = 1
            if self.use_cont_node_attr:
                # 如果使用连续节点属性，则将它们附加到one-hot编码上
                feature_onehot = np.concatenate((feature_onehot, np.array(data['attr'][i])), axis=1)
            features_onehot.append(feature_onehot)

        # 其他数据处理步骤
        if self.use_cont_node_attr:
            # 如果使用连续节点属性，更新特征数量
            num_features = features_onehot[0].shape[1]

        shapes = [len(adj) for adj in data['adj_list']]  # 计算每个图的节点数
        labels = data['targets']  # 获取图的标签
        labels -= np.min(labels)  # 将标签平移到0开始

        classes = np.unique(labels)  # 获取不同的标签类别
        num_classes = len(classes)  # 计算类别数

        if not np.all(np.diff(classes) == 1):
            # 确保标签是连续的，否则转换它们以防止PyTorch崩溃
            print('making labels sequential, otherwise pytorch might crash')
            labels_new = np.zeros(labels.shape, dtype=labels.dtype) - 1
            for lbl in range(num_classes):
                labels_new[labels == classes[lbl]] = lbl
            labels = labels_new
            classes = np.unique(labels)
            assert len(np.unique(labels)) == num_classes, np.unique(labels)

        for lbl in classes:
            # 打印每个类别的样本数量
            print('Class %d: \t\t\t%d samples' % (lbl, np.sum(labels == lbl)))

        for u in np.unique(features_all):
            # 打印每种特征值的统计信息
            print('feature {}, count {}/{}'.format(u, np.count_nonzero(features_all == u), len(features_all)))

        N_graphs = len(labels) # 计算图的总数
        assert N_graphs == len(data['adj_list']) == len(features_onehot), 'invalid training_data'

        # 创建测试集
        train_ids, test_ids = split_ids(rnd_state.permutation(N_graphs), folds=folds)

        # 创建训练集
        splits = []
        for fold in range(folds):
            splits.append({'train': train_ids[fold], 'test': test_ids[fold]})

        # 将数据添加到data字典
        data['features_onehot'] = features_onehot
        data['targets'] = labels
        data['splits'] = splits
        data['N_nodes_max'] = np.max(shapes)  # 获取最大节点数
        data['num_features'] = num_features
        data['num_classes'] = num_classes
        self.data = data   # 保存处理好的数据

    def parse_txt_file(self, fpath, line_parse_fn=None):
        """
        解析文本文件
        :param fpath: 文件路径
        :param line_parse_fn: 行解析函数
        :return: 解析后的数据
        """
        with open(pjoin(self.data_dir, fpath), 'r') as f:
            lines = f.readlines()
        data = [line_parse_fn(s) if line_parse_fn is not None else s for s in lines]
        return data

    def read_graph_adj(self, fpath, nodes, graphs):
        """
        读取图的邻接矩阵
        :param fpath: 文件路径
        :param nodes: 节点映射
        :param graphs: 图映射
        :return: 邻接矩阵列表
        """
        # print(fpath)
        # print(nodes)
        # print(graphs)
        # print("---done---")
        edges = self.parse_txt_file(fpath, line_parse_fn=lambda s: s.split(','))
        adj_dict = {}
        # print(edges)
        for edge in edges:
            node1 = int(edge[0].strip()) - 1  # -1 because of zero-indexing in our code
            node2 = int(edge[1].strip()) - 1
            # print(node1,node2)
            graph_id = nodes[node1]
            # print(nodes)
            # print(graph_id)
            assert graph_id == nodes[node2], ('invalid training_data', graph_id, nodes[node2])
            if graph_id not in adj_dict:
                n = len(graphs[graph_id])
                adj_dict[graph_id] = np.zeros((n, n))
            ind1 = np.where(graphs[graph_id] == node1)[0]
            ind2 = np.where(graphs[graph_id] == node2)[0]
            # print(ind1,ind2)
            assert len(ind1) == len(ind2) == 1, (ind1, ind2)
            adj_dict[graph_id][ind1, ind2] = 1
        adj_list = [adj_dict[graph_id] for graph_id in sorted(list(graphs.keys()))]
        return adj_list

    def read_graph_nodes_relations(self, fpath):
        """
        读取图节点和它们的关系
        :param fpath: 文件路径
        :return: 节点关系数据
        """
        graph_ids = self.parse_txt_file(fpath, line_parse_fn=lambda s: int(s.rstrip()))
        nodes, graphs = {}, {}
        for node_id, graph_id in enumerate(graph_ids):
            if graph_id not in graphs:
                graphs[graph_id] = []
            graphs[graph_id].append(node_id)
            nodes[node_id] = graph_id
        graph_ids = np.unique(list(graphs.keys()))
        unique_id = graph_ids
        for graph_id in graph_ids:
            graphs[graph_id] = np.array(graphs[graph_id])
        return nodes, graphs, unique_id

    def read_node_features(self, fpath, nodes, graphs, fn):
        """
        读取节点特征
        :param fpath: 文件路径
        :param nodes: 节点映射
        :param graphs: 图映射
        :param fn: 特征解析函数
        :return: 节点特征列表
        """
        # 读取和处理节点特征数据
        node_features_all = self.parse_txt_file(fpath, line_parse_fn=fn)
        node_features = {}
        for node_id, x in enumerate(node_features_all):
            graph_id = nodes[node_id]
            if graph_id not in node_features:
                node_features[graph_id] = [None] * len(graphs[graph_id])
            ind = np.where(graphs[graph_id] == node_id)[0]
            assert len(ind) == 1, ind
            assert node_features[graph_id][ind[0]] is None, node_features[graph_id][ind[0]]
            node_features[graph_id][ind[0]] = x
        node_features_lst = [node_features[graph_id] for graph_id in sorted(list(graphs.keys()))]
        return node_features_lst

# 创建DataReader实例并加载数据
datareader = DataReader(data_dir='./training_data/%s/' % args.dataset, rnd_state=rnd_state,
                        use_cont_node_attr=args.use_cont_node_attr, folds=args.folds)

# train and test
# 训练和测试模型
result_folds = []
all_confusion_matrices = []
# 循环遍历每个折叠
for fold_id in range(n_folds):
    """
    对每个数据折叠进行训练和测试
    :param fold_id: 当前折叠的索引
    """
    # 设置数据加载器
    loaders = []
    for split in ['train', 'test']:
        gdata = GraphData(fold_id=fold_id, datareader=datareader, split=split)
        loader = DataLoader(gdata, batch_size=args.batch_size, shuffle=split.find('train') >= 0,
                            num_workers=args.threads, collate_fn=collate_batch)
        loaders.append(loader)
    print('FOLD {}, train {}, test {}'.format(fold_id, len(loaders[0].dataset), len(loaders[1].dataset)))

    # 根据参数选择和初始化模型
    if args.model == 'gcn_modify':
        model = GCN_MODIFY(in_features=loaders[0].dataset.num_features,
                           out_features=loaders[0].dataset.num_classes,
                           n_hidden=args.n_hidden,
                           filters=args.filters,
                           dropout=args.dropout,
                           adj_sq=args.adj_sq,
                           scale_identity=args.scale_identity).to(args.device)
    elif args.model == 'gcn_origin':
        model = GCN_ORIGIN(n_feature=loaders[0].dataset.num_features,
                           n_hidden=64,
                           n_class=loaders[0].dataset.num_classes,
                           dropout=args.dropout).to(args.device)
    elif args.model == 'gat':
        model = GAT(nfeat=loaders[0].dataset.num_features,
                    nhid=64,
                    nclass=loaders[0].dataset.num_classes,
                    dropout=args.dropout,
                    alpha=args.alpha,
                    nheads=args.multi_head).to(args.device)
    else:
        raise NotImplementedError(args.model)

    print('Initialize model...')
    # 训练模型参数设置
    train_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    print('N trainable parameters:', np.sum([p.numel() for p in train_params]))
    optimizer = optim.Adam(train_params, lr=args.lr, betas=(0.5, 0.999), weight_decay=args.wd)
    # 动态调整学习率
    scheduler = lr_scheduler.MultiStepLR(optimizer, args.lr_decay_steps, gamma=0.1)  # dynamic adjustment lr
    # loss_fn = F.nll_loss  # when model is gcn_origin or gat, use this
    # 设置损失函数为交叉熵
    loss_fn = F.cross_entropy  # when model is gcn_modify, use this

    # 定义训练函数
    def train(train_loader):
        # 调整学习率。通常在每个训练周期的开始进行。
        scheduler.step()
        model.train()
        start = time.time()
        train_loss, n_samples = 0, 0
        # 通过迭代 train_loader 中的数据批次，对模型进行训练。
        for batch_idx, data in enumerate(train_loader):
            # 在每个批次中：
            for i in range(len(data)):
                # 数据被转移到设备（如 GPU）。
                data[i] = data[i].to(args.device)
            # 梯度被清零：optimizer.zero_grad()。
            optimizer.zero_grad()
            # output = model(training_data[0], training_data[1])  # when model is gcn_origin or gat, use this
            if args.model == 'gcn_modify':
                # 进行前向传播：output = model(data)。
                output = model(data)  # when model is gcn_modify, use this
            else:
                print(data)
                exit()
                # output = model(training_data[0], training_data[1])  # when model is gcn_origin or gat, use this
                output = model(data[0], data[1])  # when model is gcn_origin or gat, use this
            # 计算损失：loss = loss_fn(output, data[4])。
            loss = loss_fn(output, data[4])
            # 反向传播
            loss.backward()
            # 更新模型参数
            optimizer.step()
            time_iter = time.time() - start
            train_loss += loss.item() * len(output)
            n_samples += len(output)

        print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} (avg: {:.6f})  sec/iter: {:.4f}'.format(
            epoch + 1, n_samples, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader),
            loss.item(), train_loss / n_samples, time_iter / (batch_idx + 1)))
        # torch.save(model, 'Smartcheck.pth')

    # 定义测试函数
    def test(test_loader):
        model.eval()
        start = time.time()
        test_loss, n_samples, count = 0, 0, 0
        # tn, fp, fn, tp = 0, 0, 0, 0  # calculate recall, precision, F1 score
        accuracy, recall, precision, F1 = 0, 0, 0, 0
        # fn_list = []  # Store the contract id corresponding to the fn
        # fp_list = []  # Store the contract id corresponding to the fp
        confusion_matrix = np.zeros((loaders[0].dataset.num_classes, loaders[0].dataset.num_classes), dtype=int)
        id_list = {i: [] for i in range(loaders[0].dataset.num_classes)}

        for batch_idx, data in enumerate(test_loader):
            for i in range(len(data)):
                data[i] = data[i].to(args.device)
            # output = model(training_data[0], training_data[1])  # when model is gcn_origin or gat, use this
            output = model(data)  # when model is gcn_modify, use this
            loss = loss_fn(output, data[4], reduction='sum')
            test_loss += loss.item()
            n_samples += len(output)
            count += 1
            pred = output.detach().cpu().max(1, keepdim=True)[1]

            for k in range(len(pred)):
                # if (np.array(pred.view_as(data[4])[k]).tolist() == 1) & (
                #         np.array(data[4].detach().cpu()[k]).tolist() == 1):
                #     # TP predict == 1 & label == 1
                #     tp += 1
                #     continue
                # elif (np.array(pred.view_as(data[4])[k]).tolist() == 0) & (
                #         np.array(data[4].detach().cpu()[k]).tolist() == 0):
                #     # TN predict == 0 & label == 0
                #     tn += 1
                #     continue
                # elif (np.array(pred.view_as(data[4])[k]).tolist() == 0) & (
                #         np.array(data[4].detach().cpu()[k]).tolist() == 1):
                #     # FN predict == 0 & label == 1
                #     fn += 1
                #     fn_list.append(np.array(data[5].detach().cpu()[k]).tolist())
                #     continue
                # elif (np.array(pred.view_as(data[4])[k]).tolist() == 1) & (
                #         np.array(data[4].detach().cpu()[k]).tolist() == 0):
                #     # FP predict == 1 & label == 0
                #     fp += 1
                #     fp_list.append(np.array(data[5].detach().cpu()[k]).tolist())
                #     continue
                actual = np.array(data[4].detach().cpu()[k]).tolist()
                predicted = np.array(pred.view_as(data[4])[k]).tolist()
                confusion_matrix[actual][predicted] += 1
                if actual != predicted:
                    id_list[actual].append(np.array(data[5].detach().cpu()[k]).tolist())

            accuracy += metrics.accuracy_score(data[4], pred.view_as(data[4]))
            recall += metrics.recall_score(data[4], pred.view_as(data[4]), average='macro')
            precision += metrics.precision_score(data[4], pred.view_as(data[4]), average='macro')
            F1 += metrics.f1_score(data[4], pred.view_as(data[4]), average='macro')
            # cr = metrics.classification_report(data[4], pred.view_as(data[4]))
            # print(cr)
            # exit()
        # print(tp, fp, tn, fn)
        accuracy = 100. * accuracy / count
        recall = 100. * recall / count
        precision = 100. * precision / count
        F1 = 100. * F1 / count
        # FPR = fp / (fp + tn)

        # print(
        #     'Test set (epoch {}): Average loss: {:.4f}, Accuracy: ({:.2f}%), Recall: ({:.2f}%), Precision: ({:.2f}%), '
        #     'F1-Score: ({:.2f}%), FPR: ({:.2f}%)  sec/iter: {:.4f}\n'.format(
        #         epoch + 1, test_loss / n_samples, accuracy, recall, precision, F1, FPR,
        #         (time.time() - start) / len(test_loader))
        # )

        # print("fn_list(predict == 0 & label == 1):", fn_list)
        # print("fp_list(predict == 1 & label == 0):", fp_list)
        # print()

        print(
            'Test set (epoch {}): Average loss: {:.4f}, Accuracy: ({:.2f}%), Recall: ({:.2f}%), Precision: ({:.2f}%), '
            'F1-Score: ({:.2f}%) sec/iter: {:.4f}\n'.format(
                epoch + 1, test_loss / n_samples, accuracy, recall, precision, F1,
                (time.time() - start) / len(test_loader))
        )

        print("Confusion matrix:\n", confusion_matrix)
        print("Misclassified samples per class (id_list):", id_list)
        print()

        return accuracy, recall, precision, F1, confusion_matrix


    for epoch in range(args.epochs):
        train(loaders[0])
    accuracy, recall, precision, F1, confusion_matrix = test(loaders[1])
    result_folds.append([accuracy, recall, precision, F1])
    all_confusion_matrices.append(confusion_matrix)

print(result_folds)
acc_list = []
recall_list = []
precision_list = []
F1_list = []
# FPR_list = []

for i in range(len(result_folds)):
    acc_list.append(result_folds[i][0])
    recall_list.append(result_folds[i][1])
    precision_list.append(result_folds[i][2])
    F1_list.append(result_folds[i][3])
    # FPR_list.append(result_folds[i][4])

print(
    '{}-fold cross validation avg acc (+- std): {}% ({}%), recall (+- std): {}% ({}%), precision (+- std): {}% ({}%), '
    'F1-Score (+- std): {}% ({}%))'.format(
        n_folds, np.mean(acc_list), np.std(acc_list), np.mean(recall_list), np.std(recall_list),
        np.mean(precision_list), np.std(precision_list), np.mean(F1_list), np.std(F1_list)
    )
)

# 计算每个类别的性能指标
# 初始化存储每个类别的精度、召回率和F1得分的列表
# precision_list = []
# recall_list = []
# f1_list = []
num_classes = 3
precision_list = {class_index: [] for class_index in range(num_classes)}
recall_list = {class_index: [] for class_index in range(num_classes)}
f1_list = {class_index: [] for class_index in range(num_classes)}

for cm in all_confusion_matrices:
    # 计算每个类别的指标
    for class_index in range(num_classes):  # 假设 num_classes 是类别的数量
        # print(class_index, cm[class_index, class_index])
        TP = cm[class_index, class_index]
        FP = sum(cm[:, class_index]) - TP
        FN = sum(cm[class_index, :]) - TP

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # precision_list.append(precision)
        # recall_list.append(recall)
        # f1_list.append(f1)
        # 将结果添加到相应的列表
        precision_list[class_index].append(precision)
        recall_list[class_index].append(recall)
        f1_list[class_index].append(f1)

# print(precision_list)
# print(recall_list)
# print(f1_list)

avg_precision = {}
avg_recall = {}
avg_f1 = {}

for class_index in range(num_classes):
    avg_precision[class_index] = sum(precision_list[class_index]) / len(precision_list[class_index])
    avg_recall[class_index] = sum(recall_list[class_index]) / len(recall_list[class_index])
    avg_f1[class_index] = sum(f1_list[class_index]) / len(f1_list[class_index])

# 输出结果
print("Average Precision per class: ", avg_precision)
print("Average Recall per class: ", avg_recall)
print("Average F1-Score per class: ", avg_f1)
