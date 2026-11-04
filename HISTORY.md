# 研究の履歴

このリポジトリでの研究の経緯をまとめたもの。各ディレクトリの位置づけは [CLAUDE.md](CLAUDE.md) を参照。

## モデルの系譜

```
lite ─▶ mc_mhlf ─▶ mc_mhlite ─┬─▶ pmc_mhlite
                               └─▶ mc2_mhlite ─▶ pmc2_mhlite
```

`mc-mhef`, `mc-mhelite`, `s_pmc_mhlite`, `s_pmc2_mhlite`, `reservoir` 系は `wip/` 内で検証中の派生形。

## タイムライン

- **2026-01-28** 初回アップロード。LITEベースラインの実装と、MACDヒストグラムを導入したLITE派生の追加。
- **2026-04-03** 論文執筆用プログラムの追加、実験データ・プログラムの移行（学部の実験としてまとめ直し）、
  ディレクトリ構成の修正。
- **2026-04-04** データセット（UCR_TS_Archive_2015）のコミット対象を非対称に修正（＝データセットをgit管理から外す対応）。
- **2026-04-05** モデル比較の実施とアップロード（MC-MHLITE系の比較）、論文の改訂、参考文献の追加。
  → `experiments/undergraduate` 配下の検証・分析一式に相当。
- **2026-04-22** 論文の一部を削除する修正（MCMの説明に関する変更）。
- **2026-11-04** 直近の進捗（`wip/`）をアップロード。`model_times`（実行時間比較）をmainにマージ。
  同日、以下のリポジトリ整理を実施：
  - `experiment/` → `experiments/`、`progress/` → `wip/`、`edit/` → `papers/ieee_smc2026/`、
    `UCR_TS_Archive_2015/` → `data/UCR_TS_Archive_2015/` にリネーム。
  - 各実験配下の `valid/` → `runs/`、`performance/` → `analysis/` に統一。
  - データセットごとに量産される実行結果（`model.h5`/`training.log`、約1900ファイル）が
    誤ってgitに追跡されていたため `git rm --cached` で解消（ディスク上のファイルは保持）。
  - 実験結果ファイルを `chmod 444` にし、`.claude/` にPreToolUseフックを設定して、
    Claude Codeが誤ってこれらを削除・上書きしないよう保護。

## 論文・提出先の状況

- **IEEE SMC Hiroshima 2026**（`experiments/ieee_smc2026/`, `papers/ieee_smc2026/`）：
  MC2-MHLITE / PMC2-MHLITE 系の検証一式と論文ドラフト（`j-draft-v7.tex` ほか）が存在。
- **IIAI AAI 2026**（`experiments/iiai_aai2026/`）：LITE / MC-MHLF / MC-MHLITE / PMC-MHLITE の
  実行時間比較（`performance/time/`）を含む検証一式。
- **学部の実験**（`experiments/undergraduate/`）：最初期実装（`runs/origins`）→検証本体
  （`runs/main`）→追加検証（`runs/further`：MC-L-LITE, MC-S-LITE）の順で発展。
- **修士の実験**（`experiments/masters/`）：未着手（空フォルダ）。
- **直近の作業中**（`wip/`）：reservoir系、mc-mhef、mc-mhelite、s_pmc_mhlite、s_pmc2_mhliteなど、
  まだ論文単位に整理していない検証コード。

## 過去に破棄した取り組み（`archive/`）

- `aeon_demo`：aeonライブラリのLITE実装を用いたデモ検証。
- `mc_mhlf_fam`：MC-MHLFの派生（DWSC, FasterNet等）の初期検証。現在は後継モデルに置き換え。
