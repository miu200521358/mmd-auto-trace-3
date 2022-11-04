import random

import numpy as np


class KalmanFilter:
    def __init__(self, xvalues, yvalues, zvalues) -> None:
        self.xvalues = xvalues
        self.yvalues = yvalues
        self.zvalues = zvalues

        self.dt = 1 / 30 #計測間隔

        self.x = np.array([[self.xvalues[0]], [self.yvalues[0]], [self.zvalues[0]], [0.], [0.], [0.]]) # 初期位置と初期速度
        self.u = np.array([[0.], [0.], [0.], [0.], [0.], [0.]]) # 外部要素

        self.P = np.array([[0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.], [0., 0., 0., 1., 0., 0.], [0., 0., 0., 0., 1., 0.], [0., 0., 0., 0., 0., 1.]]) # 共分散行列
        self.F = np.array([[1., 0., 0., self.dt, 0., 0.], [0., 1., 0., 0., self.dt, 0.], [0., 0., 1., 0., 0., self.dt], [0., 0., 0., 1., 0., 0.], [0., 0., 0., 0., 1., 0.], [0., 0., 0., 0., 0., 1.]])  # 状態遷移行列
        self.H = np.array([[1., 0., 0, 0, 0., 0], [0., 1., 0., 0., 0., 0], [0., 0., 1., 0., 0., 0.]])  # 観測行列
        self.R = np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]]) #ノイズ
        self.I = np.identity((len(self.x)))    # 単位行列

    def simulate(self, n):
        # 予測
        self.x = np.dot(self.F, self.x) + self.u
        self.P = np.dot(np.dot(self.F, self.P), self.F.T)

        # 計測更新
        self.Z = np.array([self.xvalues[n], self.yvalues[n], self.zvalues[n]])
        self.y = self.Z.T - np.dot(self.H, self.x)
        self.S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        self.K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(self.S))
        self.x = self.x + np.dot(self.K, self.y)        
        self.P = np.dot((self.I - np.dot(self.K, self.H)), self.P)

        return self.x


# # https://www.yasutomo57jp.com/2021/07/22/%E3%83%91%E3%83%BC%E3%83%86%E3%82%A3%E3%82%AF%E3%83%AB%E3%83%95%E3%82%A3%E3%83%AB%E3%82%BF%E3%81%AEpython%E5%AE%9F%E8%A3%85/
# class TransitionModel(object):
#     def __init__(self):
#         pass

#     def predict(self, current):
#         # 単純に分散2の範囲で移動するモデル
#         s = current.shape
#         return current + np.random.randn(s[0], s[1]) * 2


# class ObservationModel(object):
#     def __init__(self):
#         pass

#     def likelihood(self, predictions, observation):
#         # 単純に，差の距離の逆数を尤度とするモデル
#         return 1 / np.linalg.norm(predictions - observation, ord=2, axis=1)


# class ParticleFilter(object):
#     def __init__(self, num_particle, n_dims, trans, observ, initial=None):
#         self.num_particle = num_particle
#         self.trans = trans
#         self.observer = observ

#         if initial is None:
#             self.particles = np.zeros((num_particle, n_dims))
#         else:
#             self.particles = initial

#     def update(self, obs):
#         # 前フレームからの予測
#         predictions = self.trans.predict(self.particles)
#         # ic(predictions)

#         # 予測がどれだけ現在の観測に合致しているかの評価
#         likelihoods = self.observer.likelihood(predictions, obs)
#         # ic(likelihoods)

#         # 尤度に応じてリサンプリング
#         self.resampling(predictions, likelihoods)

#         # likelihoodsで重み付けられたpredctionsの平均をとる
#         return np.average(predictions, weights=likelihoods, axis=0)

#     def resampling(self, predictions, likelihoods):
#         # 正規化した累積和を計算（全部足して1になるように）
#         slikelihoods = np.cumsum(likelihoods) / np.sum(likelihoods)
#         self.particles = np.array(random.choices(predictions, cum_weights=slikelihoods, k=self.num_particle))



# class ParticleFilter(object):
#     def __init__(self, y, n_particle, sigma_2, alpha_2):
#         self.y = y
#         self.n_particle = n_particle
#         self.sigma_2 = sigma_2
#         self.alpha_2 = alpha_2
#         self.log_likelihood = -np.inf
    
#     def norm_likelihood(self, y, x, s2):
#         return (np.sqrt(2*np.pi*s2))**(-1) * np.exp(-(y-x)**2/(2*s2))

#     def F_inv(self, w_cumsum, idx, u):
#             if np.any(w_cumsum < u) == False:
#                 return 0
#             k = np.max(idx[w_cumsum < u])
#             return k+1
        
#     def resampling(self, weights):
#         w_cumsum = np.cumsum(weights)
#         idx = np.asanyarray(range(self.n_particle))
#         k_list = np.zeros(self.n_particle, dtype=np.int32) # サンプリングしたkのリスト格納場所
        
#         # 一様分布から重みに応じてリサンプリングする添え字を取得
#         for i, u in enumerate(rd.uniform(0, 1, size=self.n_particle)):
#             k = self.F_inv(w_cumsum, idx, u)
#             k_list[i] = k
#         return k_list

#     def resampling2(self, weights):
#         """
#         計算量の少ない層化サンプリング
#         """
#         idx = np.asanyarray(range(self.n_particle))
#         u0 = rd.uniform(0, 1/self.n_particle)
#         u = [1/self.n_particle*i + u0 for i in range(self.n_particle)]
#         w_cumsum = np.cumsum(weights)
#         k = np.asanyarray([self.F_inv(w_cumsum, idx, val) for val in u])
#         return k
    
#     def simulate(self, seed=71):
#         rd.seed(seed)

#         # 時系列データ数
#         T = len(self.y)
        
#         # 潜在変数
#         x = np.zeros((T+1, self.n_particle))
#         x_resampled = np.zeros((T+1, self.n_particle))
        
#         # 潜在変数の初期値
#         initial_x = rd.normal(0, 1, size=self.n_particle)
#         x_resampled[0] = initial_x
#         x[0] = initial_x

#         # 重み
#         w        = np.zeros((T, self.n_particle))
#         w_normed = np.zeros((T, self.n_particle))

#         l = np.zeros(T) # 時刻毎の尤度

#         for t in range(T):
#             for i in range(self.n_particle):
#                 # 1階差分トレンドを適用
#                 v = rd.normal(0, np.sqrt(self.alpha_2*self.sigma_2)) # System Noise
#                 x[t+1, i] = x_resampled[t, i] + v # システムノイズの付加
#                 w[t, i] = self.norm_likelihood(self.y[t], x[t+1, i], self.sigma_2) # y[t]に対する各粒子の尤度
#             w_normed[t] = w[t]/np.sum(w[t]) # 規格化
#             l[t] = np.log(np.sum(w[t])) # 各時刻対数尤度

#             # Resampling
#             #k = self.resampling(w_normed[t]) # リサンプルで取得した粒子の添字
#             k = self.resampling2(w_normed[t]) # リサンプルで取得した粒子の添字（層化サンプリング）
#             x_resampled[t+1] = x[t+1, k]
            
#         # 全体の対数尤度
#         self.log_likelihood = np.sum(l) - T*np.log(self.n_particle)
        
#         self.x = x
#         self.x_resampled = x_resampled
#         self.w = w
#         self.w_normed = w_normed
#         self.l = l
        
#     def get_filtered_value(self):
#         """
#         尤度の重みで加重平均した値でフィルタリングされ値を算出
#         """
#         return np.diag(np.dot(self.w_normed, self.x[1:].T))
        