# data/

学習・検証に使う時系列データセットを置くディレクトリ。内容はgit管理の対象外（`.gitignore`参照）。

## UCR_TS_Archive_2015

本リポジトリのスクリプトは `data/UCR_TS_Archive_2015/` を前提にしている
（スクリプト実行時はこのフォルダが見える位置に `cd` するか、シンボリックリンクを張る。詳細は
[README.md](../README.md) の「スクリプトの実行に関する注意」を参照）。

新しく環境を構築する場合は、UCR Time Series Classification Archive (2015年版) を入手し、
`data/UCR_TS_Archive_2015/UCR_TS_Archive_2015/<データセット名>/<データセット名>_{TRAIN,TEST}` という
構造になるように配置する。
