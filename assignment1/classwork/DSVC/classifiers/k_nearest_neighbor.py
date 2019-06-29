import numpy as np
from collections import Counter

class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just 
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].  
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):#self指类实例对象本身，是定义类方法时必须有的，但不必传入参数
       
        num_test = X.shape[0]#knn文件中传入的参数X_test,所以这里是X_test中矩阵的形状
        num_train = self.X_train.shape[0]#加了self这里指类自己的X_train,类中有个X_train赋值
        dists = np.zeros((num_test, num_train))#全0矩阵
        for i in range(num_test):
            for j in range(num_train):
                this_test, this_train = X[i], self.X_train[j] # 两个向量
                # 两个向量求差值，求平方，求和，求平方根
                this_dist = np.sqrt(sum((this_test - this_train)**2))
                dists[i][j] = this_dist
        return dists

    
     
    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]#500
        num_train = self.X_train.shape[0]#5000
        dists = np.zeros((num_test, num_train))#开空间，里面全是零（500行，5000列）
        for i in range(num_test):#500行循环
            distances = np.sqrt(np.sum(np.square(self.X_train - X[i]),axis = 1))
            dists[i, :] = distances
        return dists

    def compute_distances_no_loops(self, X):
        num_test = X.shape[0]#500
        num_train = self.X_train.shape[0]#5000
        dists = np.zeros((num_test, num_train))  #500*5000
        dists = np.multiply(np.dot(X,self.X_train.T),-2)    
        sq1 = np.sum(X**2,axis=1,keepdims = True)  
        sq2 = np.sum(self.X_train**2,axis=1)
        dists = np.add(dists,sq1)  
        dists = np.add(dists,sq2)  
        dists = np.sqrt(dists)
        return dists

    def predict_labels(self, dists, k=1):  # 预测传入距离矩阵，给一个k#
        num_test = dists.shape[0]  # 500
        y_pred = np.zeros(num_test)  # 500，分类值eg car
        for i in range(num_test):
            closest_y = []
        # 第i个测试数据与所有训练样本的距离排序
        # argsort函数是将x中的元素从小到大排列，提取其对应的索引输出到y
            this_dist = dists[i, :]  # 取出当前测试向量的5000个距离
            closest_index = np.argsort(dists[i, :])[:k]  # 求出距离从小到大的k个索引
            closest_y = self.y_train[closest_index]  # 求出这些索引对应的标签值
            # 相同记数
            # # most_common(i)eg.i=1 取得[car 3]里的第一个值
            # a = Counter(closest_y)
            # y_pred[i] = a.most_common(1)[0][0]
            vote = Counter(closest_y).most_common(1)[0] #使用Counter统计出k个投票中最多的标签
            label, count = vote # label是最多的标签，count是该标签被投票的次数
            y_pred[i] = label # 当前预测的结果就是投票次数最多的标签label
        return y_pred

