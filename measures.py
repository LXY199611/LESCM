import numpy as np 
import math
import torch
import pickle
import random


def setup_seed(seed):
    # 设置随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def chebyshev(rd,pd):
    temp = np.abs(rd-pd)
    temp = np.max(temp,1)
    distance = np.mean(temp)
    return distance

def clark(rd,pd):
    temp1 = (pd - rd)**2
    temp2 = (pd + rd)**2
    temp = np.sqrt(np.sum(temp1 / temp2, 1))
    distance = np.mean(temp)
    return distance

def canberra(rd,pd):
    temp1 = np.abs(rd-pd)
    temp2 = rd + pd
    temp = np.sum(temp1 / temp2,1)
    distance = np.mean(temp)
    return distance

def kl_dist(rd,pd):
    eps = 1e-12
    temp = rd * np.log(rd / pd + eps)
    temp = np.sum(temp,1)
    distance = np.mean(temp)
    return distance

def cosine(rd,pd):
    rd = np.array(rd)
    pd = np.array(pd)
    inner = np.sum(pd*rd,1)
    temp1 = np.sqrt(np.sum(pd**2,1))
    temp2 = np.sqrt(np.sum(rd**2,1))
    temp = inner / (temp1*temp2)
    distance = np.mean(temp)
    return distance


def intersection(rd,pd):
    rd = np.array(rd)
    pd = np.array(pd)
    (rows,cols) = np.shape(rd)
    dist = np.zeros(rows)
    for i in range(rows):
        for j in range(cols):
            dist[i] = dist[i] + min(rd[i,j],pd[i,j])
    distance = np.mean(dist)
    return distance

def hamming_loss(pre_labels, test_target):

    # 获取标签的维度
    test_target[test_target==0]=-1

    num_class, num_instance = pre_labels.shape
    
    # 计算预测标签与真实标签不相同的个数
    miss_pairs = np.sum(pre_labels != test_target)
    
    # 计算 Hamming loss
    hamming_loss_value = miss_pairs / (num_class * num_instance)
    
    return hamming_loss_value

def average_precision(y_scores, y_true):

    y_true[y_true==0]=-1
    n = y_true.shape[0]  # Number of instances
    ap = 0
    
    for i in range(n):

        Y_i = np.where(y_true[i] == 1)[0]  # True positive labels for instance i
        scores_i = y_scores[i]  # Predicted scores for instance i
        
        # Rank the scores in descending order, and get corresponding label indices
        ranked_indices = np.argsort(scores_i)[::-1]
        
        sum_precision = 0
        for y in Y_i:
            # Get the rank of the current label y
            tau_i_y = np.where(ranked_indices == y)[0][0] + 1
            
            # Calculate the number of labels with score less than or equal to the current one
            correct_labels = [y_prime for y_prime in Y_i if np.where(ranked_indices == y_prime)[0][0] + 1 <= tau_i_y]
            
            # Calculate precision for this label
            precision = len(correct_labels) / tau_i_y
            sum_precision += precision
        
        # Add the average precision for this instance
        try:
            ap += sum_precision / len(Y_i)
        except:
            continue

    
    # Return the average over all instances
    return ap / n

def one_error(outputs, test_target):

    test_target[test_target==0] = -1

    num_class, num_instance = outputs.shape
    temp_outputs = []
    temp_test_target = []

    # Remove instances where all labels are either 1 or -1
    for i in range(num_instance):
        temp = test_target[:, i]
        if (np.sum(temp) != num_class) and (np.sum(temp) != -num_class):
            temp_outputs.append(outputs[:, i])
            temp_test_target.append(temp)
    
    outputs = np.array(temp_outputs).T
    test_target = np.array(temp_test_target).T
    num_class, num_instance = outputs.shape
    
    # Initialize labels
    Label = [set() for _ in range(num_instance)]
    Label_size = np.zeros(num_instance)

    for i in range(num_instance):
        temp = test_target[:, i]
        Label_size[i] = np.sum(temp == 1)
        for j in range(num_class):
            if temp[j] == 1:
                Label[i].add(j)

    # Compute One Error
    oneerr = 0
    for i in range(num_instance):
        indicator = 0
        temp = outputs[:, i]
        max_value = np.max(temp)
        
        for j in range(num_class):
            if temp[j] == max_value:
                if j in Label[i]:
                    indicator = 1
                    break

        if indicator == 0:
            oneerr += 1

    return oneerr / num_instance


def ranking_loss(outputs, test_target):

    test_target[test_target==0] = -1
    
    num_class, num_instance = outputs.shape
    temp_outputs = []
    temp_test_target = []

    # Remove instances where all labels are either 1 or -1
    for i in range(num_instance):
        temp = test_target[:, i]
        if (np.sum(temp) != num_class) and (np.sum(temp) != -num_class):
            temp_outputs.append(outputs[:, i])
            temp_test_target.append(temp)
    
    outputs = np.array(temp_outputs).T
    test_target = np.array(temp_test_target).T
    num_class, num_instance = outputs.shape
    
    # Initialize labels
    Label = []
    not_Label = []
    Label_size = np.zeros(num_instance)

    for i in range(num_instance):
        temp = test_target[:, i]
        Label_i = np.where(temp == 1)[0]  # Find indices where label is 1
        not_Label_i = np.where(temp == -1)[0]  # Find indices where label is -1
        Label.append(Label_i)
        not_Label.append(not_Label_i)
        Label_size[i] = len(Label_i)

    # Compute Ranking Loss
    rankloss = 0

    for i in range(num_instance):
        temp = 0
        for m in range(int(Label_size[i])):
            for n in range(num_class - int(Label_size[i])):
                if outputs[Label[i][m], i] <= outputs[not_Label[i][n], i]:
                    temp += 1
        # Compute binary ranking loss for the instance
        rl_binary = temp / (len(Label[i]) * len(not_Label[i]))
        rankloss += rl_binary

    # Return the average ranking loss over all instances
    return rankloss / num_instance

def predict_distribution(d):
    """
    Function to predict logical labels based on the distribution
    Args:
    d (numpy array): Input array of predicted distributions
    
    Returns:
    logicallabel (numpy array): Array of logical labels (-1 or 1)
    """
    t = 0.4
    logicallabel = -1 * np.ones(d.shape)  # 初始化为 -1 的数组
    
    # 遍历每一行
    for i in range(d.shape[0]):
        # 对每一行进行降序排列，并获取对应的索引
        sorted_row_indices = np.argsort(d[i, :])[::-1]
        sorted_row = d[i, sorted_row_indices]
        
        sum_val = 0
        
        # 遍历排序后的值
        for j in range(len(sorted_row)):
            sum_val += sorted_row[j]
            logicallabel[i, sorted_row_indices[j]] = 1  # 标记为 1
            
            if sum_val >= 0.5:
                break  # 如果累加和超过 0.5，跳出循环
    
    return logicallabel

