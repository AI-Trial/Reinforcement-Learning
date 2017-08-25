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

4.4. Train CartPole with DQN
```
$ python train_dqn_gym.py --env CartPole-v0 --gpu -1
```

4.5. Load trained network
```
$ python train_dqn_gym.py --env CartPole-v0 --gpu -1 --demo --monitor --load dqn_out/保存フォルダ名/100000_finish
```

4.6. Train CartPole by DQN with prioritized replay and episodic replay  
計算時間の目安: 100,000[steps]/(1,000[steps]/75[sec])=2.1[hours]
```
$ python train_dqn_gym.py --env CartPole-v0 --gpu -1 --steps 100000 --prioritized-replay --episodic-replay
```
