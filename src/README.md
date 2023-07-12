#Author
Yoshiki Yano

## CNNによる訓練とOptunaによるハイパラメータ最適化
今回はCIFR10とMNISTにおいてCNNによる分類機の訓練とハイパラメータの最適化を同時に行なった．
詳細は
- Optuna_with_PyTorch_CIFAR10.ipynb
- Optuna_with_PyTorch_MNIST.ipynb
にて記載

jupiter notebookの起動手順は../README.mdに記載してある通りである．

## 任意パラメータと最適パラメータとの比較
NNの学習状態の検証においてある特定のハイパラメータを任意に変更し、その影響を確認したいという場面が度々発生する．
この問題は最適パラメータを作成し終えてからも発生するもので任意のパラメータ指定は学習状態検証において必須の事項となっている


今回は以下のpythonファイルにて任意パラメータの指定を実現した．(Optuna_MNIST_direct.py, Optuna_CIFAR10_direct.py)
以下のコマンドで実行可能である．

MNIST
python Optuna_MNIST_direct.py --patch_size 128 --EPOCH 10 --num_layer 6 --mid_units 400 --num_filter 128 112 128 32 64 16 --activation "ELU" --optimizer "MomentumSGD" --weight_decay 3.705858691297322e-09 --adam_lr 0.00021312 --momentum_sgd_lr 0.014335285805707738 --seed 42

CIFAR10
python Optuna_CIFAR10_direct.py --patch_size 4 --EPOCH 10 --num_layer 4 --mid_units 140 --num_filter 128 112 112 112 --activation_name "ReLU" --optimizer_name "MomentumSGD" --weight_decay 5.2182135446336915e-08 --adam_lr 0.00021312 --momentum_sgd_lr 0.0004955865902351846 --seed 42

また上記学習結果はtensorboardにて可視化できるようになっているためそのtensorboard実行コマンドも以下に示す．
tensorboard --logdir /root/src/logs/MNIST/<実行した時刻>/learning --ip 0.0.0.0 --port 6006 --allow-root
tensorboard --logdir /root/src/logs/MNIST/<実行した時刻>/learning --ip 0.0.0.0 --port 6006 --allow-root

## 結果
結果として最適パラメータでの学習結果と任意の非最適なハイパーパラメータでの学習結果を以下に示す. 
これによって変化させたパラメータが学習結果に対してどのように’寄与するのかを確認できる. 
