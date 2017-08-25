# Reinforcement-Learning

1. レポジトリのクローン

```
$ git clone https://github.com/AI-Trial/Reinforcement-Learning.git
```

2. ローカル編集前に必ず以下のコマンドで最新に反映すること。

```
$ git pull
```

3. 編集後は以下のコマンドでリモートをアップデート。

```
$ git add -A
$ git commit -m "<message>"
$ git push origin master
```
4. Chainer RLのサンプルを動かす方法
  
4.0. 依存ライブラリのインストール
```
sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
```
  
4.1. Install gym
```
$ git clone https://github.com/openai/gym.git
$ cd gym
$ sudo pip install -e .
```
  
4.2. Clone ChainerRL
```
$ cd ..
$ git clone https://github.com/chainer/chainerrl.git
```
  
4.3. Chainer RL exampleの場所
```
$ cd chainerrl/examples/gym
```
  
4.4. Train CartPole by DQN
```
$ python train_dqn_gym.py --env CartPole-v0 --gpu -1
# Load trained network  
$ python train_dqn_gym.py --env CartPole-v0 --gpu -1 --demo --monitor --load dqn_out/保存ディレクトリ名/100000_finish
```
4.5. Train CartPole by DQN with prioritized replay and episodic replay
```
# 計算時間の目安: 100,000[steps]/(1,000[steps]/38[sec])=1[hours]
$ python train_dqn_gym.py --env CartPole-v0 --gpu -1 --steps 100000 --prioritized-replay --episodic-replay
# Load trained network  
$ python train_dqn_gym.py --env CartPole-v0 --gpu -1 --demo --monitor --load dqn_out/保存ディレクトリ名/100000_finish
```

5. Box2DでBipedalWalkerを動かす
  
5.0. スタックしやすいところ  
Ubuntu 14.04のpipではswig2.0が入ってきますが、Box2Dはswig>3.0が必要です。  
参考： https://www.bountysource.com/issues/34260638-box2d-won-t-find-some-rand_limit_swigconstant
  
5.1. Install swig > 3.0
```
$ echo deb http://archive.ubuntu.com/ubuntu trusty-backports main restricted universe multiverse | sudo tee /etc/apt/sources.list.d/box2d-py-swig.list
$ sudo apt-get install -t trusty-backports swig3.0
$ sudo apt-get remove swig swig2.0
$ sudo ln -s /usr/bin/swig3.0 /usr/bin/swig
$ swig -version
```
5.2. Install Box2D from source
```
$ sudo pip uninstall Box2D box2d-py
$ git clone https://github.com/pybox2d/pybox2d
$ cd pybox2d
$ python setup.py build
$ sudo python setup.py install
```
5.3. Install gym Box2D env
```
# pipを最新版にアップデート
$ sudo pip install --ignore-installed pip
$ sudo pip install -e '.[box2d]'
```
5.4. Train BipedalWalker by DQN
```
$ python train_dqn_gym.py --env BipedalWalker-v2 --gpu -1
# Load trained network
$ python train_dqn_gym.py --env BipedalWalker-v2 --gpu -1 --demo --monitor --load dqn_out/保存ディレクトリ名/100000_finish
```
5.5. Train BipedalWalker by DQN with prioritized replay and episodic replay
```
# 計算時間の目安: 100,000[steps]/(1,000[steps]/75[sec])=2.1[hours]
$ python train_dqn_gym.py --env BipedalWalker-v2 --gpu -1 --steps 100000 --prioritized-replay --episodic-replay
# Load trained network
$ python train_dqn_gym.py --env BipedalWalker-v2 --gpu -1 --demo --monitor --load dqn_out/保存ディレクトリ名/100000_finish
```
