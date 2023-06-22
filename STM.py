import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from json import dumps
import copy
import warnings
warnings.filterwarnings("ignore")
class STM(BaseEstimator, ClassifierMixin):
    def __init__(
        self, n_domains = 4, n_class = 4, label = 'label', Cs = 1, kernels = 'poly', k_list = 1, beta_list = 0.1, gamma_list = 0.1, 
        print_acc = False, SVM = False, best_sources = False):
        
        self.DOMAINS_TRAIN = dict()
        self.n_domains = n_domains
        self.label = label 
        self.n_class = n_class
        self.print_acc = print_acc
        self.SVM = SVM
        self.sources_reduced = dict()
        self.k_list = k_list
        self.Cs = Cs
        self.kernels = kernels
        self.beta_list = beta_list
        self.gamma_list = gamma_list
        self.best_sources = best_sources
           

    def build_source_classifier(self, source_X, source_y, target_X, target_y):
        md = RandomForestClassifier().fit(source_X, source_y)
        target_score = md.score(target_X, target_y)
        return md, target_score

    def calculate_best_sources(self):
        for i, (df_x, df_y) in enumerate(zip(self.S_x, self.S_y)):
            classifier, score = self.build_source_classifier(source_X = df_x, source_y = df_y, target_X = self.T_xl, target_y = self.T_yl)
            if self.print_acc:
                print("Classifier {}: {}".format(i, score))
            self.DOMAINS_TRAIN[i].update(classifier=classifier)
            self.DOMAINS_TRAIN[i].update(score=score)
        
        if self.best_sources:
            new_dict = dict.fromkeys(self.DOMAINS_TRAIN.keys(), None)
            for i in new_dict:
                new_dict[i] = self.DOMAINS_TRAIN[i]['score']
            new_dict = {k: v for k, v in sorted(new_dict.items(), key=lambda item: item[1], reverse=True)}
            self.OLD_DICT = self.DOMAINS_TRAIN.copy()
            for i, k in enumerate(new_dict.keys()):
                if i > 3:
                    del self.DOMAINS_TRAIN[k]

       
    def reduce_sv(self):
        for d in self.DOMAINS_TRAIN:
            domain = self.DOMAINS_TRAIN[d]
            model = SVC(domain['C'], domain['kernel'], random_state=0).fit(
                self.S_x[d], self.S_y[d])

            source_reduced = self.S_x[d].drop(model.support_)
            y_reduced = self.S_y[d].drop(model.support_)
            self.sources_reduced[d] = (source_reduced, y_reduced)
    
    def find_proto(self, df_idx):
        '''
        df_idx: the index of this domain in DOMAINS
        '''
        df_x = self.sources_reduced[df_idx][0]
        df_y = self.sources_reduced[df_idx][1]
        proto_x = pd.DataFrame([], columns=df_x.columns)
        proto_y = []

        for i in range(self.n_class):
            subset = df_x[(df_y == i).values]
            # print("subset: ", subset)
            km = KMeans(n_clusters = self.DOMAINS_TRAIN[df_idx]['k'], random_state=0).fit(
                subset)
            proto_temp = pd.DataFrame(km.cluster_centers_, columns=df_x.columns)
            proto_y.extend([i] * self.DOMAINS_TRAIN[df_idx]['k'])

            proto_x = pd.concat([proto_x, proto_temp])
        
        return proto_x, proto_y

    # destination function: nearest prototype
    # np.linalg.norm default performs L2-norm on arrays and Frobenius norm on matrices
    def destination(self, x_t, y_t, source_df_x, source_df_y):
        # get all samples from class y; y in (0, 1, 2)
        X_s = source_df_x[source_df_y == y_t]
        X_s.reset_index(inplace=True, drop=True)

        # compute distance from x to all samples in class y
        dist = np.linalg.norm(x_t - X_s, axis=1)**2

        # take argmin
        min_id = np.argmin(dist)
        # print("min_id: ", min_id)
        res = X_s.iloc[min_id, :]
        return res

    def learn_STM_param(self, O, D, F, beta, gamma):
        # gamma non zero, otherwise singular matrix

        n, m = O.shape

        F_diag = np.diag(F)
        
        F_hat = n + gamma
        F_hat = np.array([F_hat] * n)
        F_hat_diag = np.diag(F_hat)

        F_hat_inv = 1/F_hat
        F_hat_inv_diag = np.diag(F_hat_inv)

        O_hat = O.T.dot(F_diag) # 2 x n
        D_hat = D.T.dot(F_diag) # 2 x n

        Q = D.T.dot(F_diag).dot(O) - D_hat.dot(F_hat_inv_diag).dot(O_hat.T) + np.diag(np.ones(m) * beta) # 2 x 2
        P = O.T.dot(F_diag).dot(O) - O_hat.dot(F_hat_inv_diag).dot(O_hat.T) + np.diag(np.ones(m) * beta)
        
        A = Q.dot(np.linalg.inv(P))
        assert A.shape == (m, m)

        b = (D_hat - A.dot(O_hat)).dot(F_hat_inv)

        assert b.shape == (m, )

        return A, b

    def supervised_STM(self, target_df_x, target_df_y, source_df_x, source_df_y, beta = 100, gamma = 10000):
        # origin df
        O = target_df_x
        Y_t = target_df_y
        # destination df
        D = pd.DataFrame([], columns = target_df_x.columns)
        for i, row in O.iterrows():
            # print("row: ", row)
            d_i = self.destination(x_t = row, y_t = Y_t[i], source_df_x = source_df_x, source_df_y = source_df_y)
            D = D.append(d_i)
        D.reset_index(inplace=True, drop=True)

        assert len(O) == len(D)
        F = np.ones(len(O))

        # learn STM A0, b0
        A0, b0 = self.learn_STM_param(O.to_numpy(), D.to_numpy(), F, beta = beta, gamma = gamma)

        return A0, b0
    
    def get_source_dfs(self, X, y):
        domains = list(X['source_no'].unique())
        domains.sort()
        # print("domains: ", domains)
        # assert self.n_domains == (len(domains) - 1)
        X_s_list = []
        y_s_list = []

        bool_t = (X['source_no'] == 0).values
        X_t = X[bool_t].reset_index(drop=True).drop('source_no', axis=1)
        y_t = y[bool_t].reset_index(drop=True)
        # print("target set: ", len(X_t), " ", len(y_t))
        for d in domains[1:]:
            bool_s = (X['source_no'] == d).values
            X_s = X[bool_s].reset_index(drop=True).drop('source_no', axis=1)
            y_s = y[bool_s].reset_index(drop=True)
            # print("source ", d, " set: ", len(X_s), " ", len(y_s))
            X_s_list.append(X_s)
            y_s_list.append(y_s)
        
        return X_s_list, y_s_list, X_t, y_t 

    def reset_params(self):
        if not isinstance(self.k_list, tuple):
            self.k_list = [self.k_list] * self.n_domains
        else:
            self.k_list = list(self.k_list)

        if self.SVM:
            if not isinstance(self.Cs, tuple):
                self.Cs = [self.Cs] * self.n_domains
            else:
                self.Cs = list(self.Cs)
            
            if not isinstance(self.kernels, tuple):
                self.kernels = [self.kernels] * self.n_domains
            else:
                self.kernels = list(self.kernels)
        else:
            self.Cs = list()
            self.kernels = list()

        if not isinstance(self.beta_list, tuple):
            self.beta_list = [self.beta_list] * self.n_domains
        else:
            self.beta_list = list(self.beta_list)

        if not isinstance(self.gamma_list, tuple):
            self.gamma_list = [self.gamma_list] * self.n_domains
        else:
            self.gamma_list = list(self.gamma_list)

        for i in range(self.n_domains):
            if self.SVM:
                self.DOMAINS_TRAIN[i] = dict(k = self.k_list[i], beta = self.beta_list[i], gamma = self.gamma_list[i], C = self.Cs[i], kernel = self.kernels[i])
            else:
                self.DOMAINS_TRAIN[i] = dict(k = self.k_list[i], beta = self.beta_list[i], gamma = self.gamma_list[i])
            

    def fit(self, X, y=None):
        self.reset_params()
        
        # extract source and target sets for training
        self.S_x, self.S_y, self.T_xl, self.T_yl = self.get_source_dfs(X, y)

        # train model on sources and predict the labeled target data
        self.calculate_best_sources()

        # start the learning 
        if self.SVM:
            self.reduce_sv()
        else:
            for d in self.DOMAINS_TRAIN:
                
                self.sources_reduced[d] = (self.S_x[d], self.S_y[d])

        for i in self.sources_reduced:
            proto_x, proto_y = self.find_proto(i)

            A, b = self.supervised_STM(
                self.T_xl, self.T_yl, source_df_x = proto_x, source_df_y = proto_y,
                beta =self.DOMAINS_TRAIN[i]['beta'], gamma = self.DOMAINS_TRAIN[i]['gamma'])
            
            self.DOMAINS_TRAIN[i].update(A = A)
            self.DOMAINS_TRAIN[i].update(b = b)
        
        return self

    
    def predict(self, X, y=None):
        X_ = X.copy()
        numerator = 0
        denom = 0
        try:
            X_.drop('source_no', axis=1, inplace=True)
        except:
            pass

        for i in self.sources_reduced:
            # transform test data
            
            X_transform = self.DOMAINS_TRAIN[i]['A'].dot(X_.T).T + self.DOMAINS_TRAIN[i]['b']
            # X_transform = X_
            # self.X_trans = X_transform
            y_ = self.DOMAINS_TRAIN[i]['classifier'].predict(X_transform)

            self.DOMAINS_TRAIN[i].update(prediction = y_)

            numerator += self.DOMAINS_TRAIN[i]['score'] * y_
            denom += self.DOMAINS_TRAIN[i]['score']


        self.weighted_preds_ = np.round(numerator/denom)
        # self.accuracy_ = np.mean(self.weighted_preds_ == y)
        return self.weighted_preds_
    
    def third_metric(self, X, y=None):
        X_ = X.copy()
        numerator = 0
        denom = 0
        try:
            X_.drop('source_no', axis=1, inplace=True)
        except:
            pass

        for i in self.sources_reduced:    
            X_transform = X_
            pred = self.DOMAINS_TRAIN[i]['classifier'].predict(X_transform)

            numerator = numerator + self.DOMAINS_TRAIN[i]['score'] * pred
            denom = denom + self.DOMAINS_TRAIN[i]['score']


        weighted_preds_ = np.round(numerator/denom)
        
        accuracy_ = np.mean(weighted_preds_ == y)
        return accuracy_
    
    def score(self, X, y= None):
        return np.mean(self.predict(X, y) == y)

    def save_model(self, file_name):
        temp_dict = copy.deepcopy(self.DOMAINS_TRAIN) 
        for i in temp_dict:
            temp_dict[i].pop('classifier')
            temp_dict[i].pop('score')
            try:
                temp_dict[i].pop('prediction')
                temp_dict[i]['chosen'] = 'T'
            except:
                pass
        
        json_file = dumps(temp_dict)
        f = open(file_name,"w")
        f.write(json_file)
        f.close()

