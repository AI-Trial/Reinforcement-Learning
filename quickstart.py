#!/usr/bin/env python
# Ubuntu 14.04, python 2.7で動作確認
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import gym
import numpy as np

env = gym.make('CartPole-v0') # CartPole-v0環境を作成
# ここを自分で作った環境にすればオリジナルの環境で学習可能
# 例えば、env=myturtlebot()とか。

print('observation space:', env.observation_space) # observation space(stateの数)を確認
# [position of cart, velocity of cart, angle of pole, rotation rate of pole] の4つ
print('action space:', env.action_space) # action space(actionの数)を確認
# [right, left] の2つ

obs = env.reset() # 環境を初期化
env.render() # 画面出力
print('initial observation:', obs) # observation spaceの初期値を確認

action = env.action_space.sample() # ランダムにactionを選択
obs, r, done, info = env.step(action) # actionを実行してtimestepを進める。

print('next observation:', obs) # action実行後の状態
print('reward:', r) # 報酬
print('done:', done) # 終了フラグ。ポールが倒れたらdone
print('info:', info) # その他の情報

class QFunction(chainer.Chain): # chainer.Chainクラスを継承してQ関数クラスを定義
    def __init__(self, obs_size, n_actions, n_hidden_channels=50):
        # obs_size: 状態数, n_actions: アクション数, n_hidden_channels: 隠れ層のチャネル数):
        super(QFunction, self).__init__( # python 2.7 だとQuick Startとちょっと記述方法が異なる。
            l0=L.Linear(obs_size, n_hidden_channels), # input layer, 4 channels
            l1=L.Linear(n_hidden_channels, n_hidden_channels), # hidden layer, 50 channels
            l2=L.Linear(n_hidden_channels, n_actions)) # output layer, 2 channels
    def __call__(self, x, test=False):
        # input xを入力した際のネットワーク出力(action)を返す関数
        h = F.tanh(self.l0(x)) # input layer 出力。活性化関数tanh
        h = F.tanh(self.l1(h)) # hidden layer 出力。活性化関数tanh
        # output layer 出力からactionを生成
        return chainerrl.action_value.DiscreteActionValue(self.l2(h))

obs_size = env.observation_space.shape[0] # observation spaceのサイズ(4)
n_actions = env.action_space.n # action spaceのサイズ(2)
q_func = QFunction(obs_size, n_actions) # Q関数インスタンスを作成

# Q関数をAdamで最適化。ε=0.01
optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)

gamma = 0.95 # 割引率

# ε-greedy, ε=0.3で固定。
explorer = chainerrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=env.action_space.sample)

# Experience Replay の replay bufferを設定
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)

# CartPole-v0 の observation space は numpy.float64
# Chainerはデフォルトでnumpy.float32しか扱えないので変換が必要
# float64 -> float32に変換するだけの関数φ
phi = lambda x: x.astype(np.float32, copy=False)

# agentを作成, Double DQNを使用
# http://musyoku.github.io/2016/03/16/deep-reinforcement-learning-with-double-q-learning/
agent = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=500, update_interval=1,
    target_update_interval=100, phi=phi)

n_episodes = 200 # 学習するEpisode数=200
max_episode_len = 200 # 1 Episodeあたりの最大timestep

for i in range(1, n_episodes + 1):
    obs = env.reset() # 環境の初期化
    reward = 0 # 報酬(reward)の初期化
    done = False
    R = 0  # 収益(return)の初期化
    t = 0  # timestep初期化
    while not done and t < max_episode_len:
        env.render() # 画面表示
        # 現在の状態(s)からaction(a)を生成し、(s,a,r)をReply Memoryに保存
        action = agent.act_and_train(obs, reward)
        obs, reward, done, _ = env.step(action) # 生成したaction(at)を実行
        R += reward # 収益更新
        t += 1 # timestep更新
    if i % 10 == 0: # ログ表示
        print('episode:', i,
              'R:', R,
              'statistics:', agent.get_statistics())
    agent.stop_episode_and_train(obs, reward, done) # Q-network更新

print('Finished.')

# 保存
agent.save('agent')
# model.npz -> Q-network
# target_model.npz -> 教師信号生成用Q-network
