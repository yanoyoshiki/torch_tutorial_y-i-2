# torch_tutorial_y_i-2
The repository for M1 tutorial. 

The administers are YANO and NAKAMURA. 

このリポジトリは「課題2: ハイパーパラメータチューニングに慣れる」に関するリポジトリです.

# 課題
<<<<<<< HEAD
以下の1.-8.を実施する．
=======
今回の課題では1.-8.の指示があった．
>>>>>>> 5783b88335c72a3539fc02ed480326d9b6e99c68

1. CIFAR-10とMNISTを落としてくる
2. Tensorboardをつかうので，知らない人は調べる
3. CNNでネットワークを作る(各種パラメータは調整するので変更が容易な形で実装)
4. MNISTを使って，ネットワークの深さ，学習率，バッチサイズなどのハイパーパラメータを変化させながら
モデルの性能を評価
5. CIFAR-10でも同様の作業を行う
6. 各ハイパーパラメータが結果にどのように影響するか比較，考察
7. どのように最適なパラメータを探すかについて調査
8. (できる人だけ) Optunaなどのハイパーパラメータチューニングライブラリによる自動化


<<<<<<< HEAD
=======
しかし今回はOptunaに慣れておきたかったため、Optunaベースで上記課題を達成した．
具体的には
1. CIFAR-10とMNISTを落としてくる
2. Optunaの使用を前提にそれぞれのCNNを組む
3. 両データセットにおいて学習及びハイパラメータの最適化を行う．
4. Tensorboardにてネットワークに関するハイパラメータがどのように結果に寄与するのか分析を行う
5. 任意パラメータの入力により最適パラメータと最適ではない任意のパラメータを入力することで精度の比較を行う

# Set up
>>>>>>> 5783b88335c72a3539fc02ed480326d9b6e99c68
# Quick Start
## SSH接続(SSH接続でリモートPCへ接続する場合のみlocal terminalにて実行が必要)
```
 ssh -L 63322:localhost:63322 -L 6006:localhost:6006 -L 6007:localhost:6007  <username>@<remotePC IP>
```
接続完了後， 環境構築へ
## nativeの場合
環境構築へ
<<<<<<< HEAD
## 環境設定
=======
## 環境構築
>>>>>>> 5783b88335c72a3539fc02ed480326d9b6e99c68
### 初回
任意のディレクトリにて， githubからリポジトリのクローンを作成する．
```
git clone https://github.com/Issa-N/torch_tutorial_y-i-3.git
```
次に，Dockerで仮想環境を作る．
```
cd torch_tutorial_y-i-3/docker

bash build.sh
bash run.sh
cd
```
### 2回目以降
以下のコマンドで，dockerのコンテナへの接続を行う．
```
docker exec -it torch_tutorial_y-i　bash
cd
```
一時的に抜ける時は 「controll + P + Q」

<<<<<<< HEAD
## Jupiter notebookを開く時
```
cd
jupyter-notebook --ip 0.0.0.0 --port 63322 --allow-root
```


=======
### Jupiter notebookを開く時
```
jupyter-notebook --ip 0.0.0.0 --port 63322 --allow-root
```

## 環境構築終了後は　src内のreadme 参照
>>>>>>> 5783b88335c72a3539fc02ed480326d9b6e99c68
