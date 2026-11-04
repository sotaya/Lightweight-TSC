# Lightweight-TSC

時系列分類（Time Series Classification）向けの軽量深層学習モデルを研究するリポジトリ。
UCR_TS_Archive_2015 データセットを用いて、LITE系モデル（LITE / MC-MHLF / MC-MHLITE / PMC-MHLITE 等）の
派生モデルを学習・検証し、学会・論文単位で結果をまとめている。

## セッション開始時に読むこと（重要）

このリポジトリで研究の相談・作業を始める際は、本ファイルの内容に加えて以下を**作業前に読むこと**。
特にユーザーから「現状どうなってる？」「続きをやりたい」のように前提共有を省略した形で話しかけられた
場合は、聞き返す前にまずこれらを読んで状況を把握する。

1. **`TODO.md`** — 現在進行中の研究タスクと直近の優先順位（最も頻繁に更新される、現状把握の一次情報源）。
2. **`RESEARCH_LOG.md`**（存在する場合。gitignore対象の個人メモなのでリポジトリ初期状態には無いことがある）
   — ゼミ発表資料を解析して再構成した研究の経緯（モデルの変遷・意思決定の背景・実験結果の数値）。
3. **`HISTORY.md`** — リポジトリのディレクトリ構成・論文投稿先の変遷（いつ何を変更したか、どの学会にどの状態で提出しているか）。

これらは役割が異なるため使い分けること：`TODO.md`は「次に何をやるか」、`RESEARCH_LOG.md`は
「なぜ今の状態に至ったか（経緯・失敗した案とその理由）」、`HISTORY.md`は「リポジトリ自体の変遷」。
作業の結果、タスクの状態や優先順位が変わったら**その場で`TODO.md`を更新する**
（完了したら`[x]`、新しい知見・次の一手は追記。検証結果の数値は`wip/analysis/`等の集計ファイルか、
実際に`wip/runs/`以下の結果ファイルを読んで確認し、思い込みで書かない）。

## ディレクトリの役割

- `data/UCR_TS_Archive_2015/` — ベンチマークデータセット（gitignore対象、.envとともに手元にのみ存在）
- `experiments/<venue>/runs/` — 各モデルの学習スクリプトと実行結果（`model.h5`, `training.log`, 実行時間csv等）。
  **過去の実行結果なので削除・上書きしないこと。**
- `experiments/<venue>/analysis/` — `runs/` の結果を集計・統計検定（Wilcoxon等）するスクリプトとcsv/png。
  こちらは自由に整理・統合してよい。
- `experiments/ieee_smc2026/`, `experiments/iiai_aai2026/` — 投稿済み/投稿予定の論文に対応する実験一式。
- `experiments/undergraduate/` — 学部の実験。`runs/origins`（最初期実装）→`runs/main`（検証本体）→`runs/further`（追加検証）の順で発展。
- `experiments/masters/` — 修士の実験用（現時点では空）。
- `wip/` — まだどの論文にも紐づいていない、直近の作業中コード。構成は `experiments/<venue>/` と同じ（`runs/` / `analysis/`）。
  ある程度まとまったら `experiments/<新しい venue名>/` として切り出す想定。
- `archive/` — 過去に方針転換・破棄したコード（aeon_demo, mc_mhlf_fam）。参照用に残しているだけで現役ではない。
- `papers/<venue>/` — 論文執筆物（tex/bib/doc/fig/pdf等）。gitignore対象で、図など一部ファイルのみ明示的に追跡。

## モデル名の対応関係

`lite` → `mc_mhlf` → `mc_mhlite` → `pmc_mhlite` / `mc2_mhlite` → `pmc2_mhlite` という順で派生・改良が進んでいる
（`mc-mhef`, `mc-mhelite`, `s_pmc_mhlite`, `reservoir` などは `wip/` 内の実験的な変種）。
同名スクリプトが `experiments/` 内の複数フォルダに存在するのは、論文ごとに実行時点のコードをスナップショットとして
残しているため（意図的な重複であり、共通化はしていない）。

## スクリプト実行時の注意（重要）

学習スクリプトはデータセットや出力フォルダを `pathlib.Path("UCR_TS_Archive_2015")` のように
**スクリプト自身の場所ではなく実行時のカレントディレクトリからの相対パス**で参照する。
そのため新しい実験を実行する・再現する際は、スクリプトのあるフォルダに `cd` し、そこから見える位置に
データセット（`data/UCR_TS_Archive_2015`）をコピーまたはシンボリックリンクしてから実行する。

## 作業時の方針

- `runs/` 配下の実行結果（`model.h5`, `training.log`, 実行時間csv等）は過去の研究成果として保持する。削除・上書き禁止。
- `analysis/` 配下の csv/Excel は再集計・統合して構わない。
- 新しい論文・実験を始めるときは `experiments/<venue>/{runs,analysis}/` の構成を踏襲する。

## 実験結果の保護（重要）

過去の実験結果（`model.h5`, `training.log`, `best_params.json`, `model.pkl`, `output*.log`）および
`runs/` ディレクトリを誤って削除・上書きしないよう、二重の保護をかけている。

1. **ファイル権限**: 既存の実験結果ファイルは `chmod 444`（読み取り専用）にしてある。
   Pythonスクリプトなどが同じパスに書き込もうとしても `PermissionError` になる。
   意図的に再生成する場合は `chmod +w` してから実行する。
2. **フック**: `.claude/settings.json` に `PreToolUse` フックを設定し、
   `.claude/hooks/guard_results_bash.sh`（`rm`/`mv`等で上記ファイルや`runs/`を操作するBashコマンドを拒否）
   と `.claude/hooks/guard_results_write.sh`（Write/Edit/NotebookEditでの上書きを拒否）で
   Claude Code自身の操作もブロックする。
   これはヒューリスティックな安全網であり、完全な保証ではない。新しい命名規則の結果ファイルが
   出てきた場合は `.claude/hooks/guard_results_bash.sh` 内の `protected` 正規表現に追記する。

## gitの操作について（重要）

`git add`/`commit`/`push`/`merge`/`rebase`/`reset`/`checkout`/`branch -d`等の変更を伴うgit操作は、
`.claude/settings.json` の `permissions.ask` で**必ず実行前に確認を求める**設定にしてある。
Claude Codeが自分の判断だけでコミット・pushをしないようにするための設定であり、
ユーザーから明示的に指示されたときのみ実行すること。
`git status`/`log`/`diff`/`show`等の読み取り専用コマンドは確認なしで実行してよい。

同様に `rm`/`rmdir`/`unlink`/`shred`/`truncate` によるファイル削除も、`permissions.ask` で
**実行前に必ず確認を求める**設定にしてある（`model.h5`等の保護対象ファイルは前述のフックで
確認すら許さず拒否する）。ただし、これはBashコマンドの先頭トークンで判定する仕組みのため、
`find ... -delete` や Pythonスクリプト内の `os.remove()` 等、`rm`で始まらない削除は捕捉できない。
そうした手段で削除する前提の操作は避け、通常は `rm` を使うこと。

## git管理の方針（重要）

データセットの数（UCRは84種）だけファイルが量産される実行結果（`model.h5`, `training.log` 等、
データセットごとのサブフォルダに1個ずつできるもの）は**gitにコミットしない**。
ディスク上には保持するが、追跡対象には加えない。

- `.gitignore` に `model.h5` / `training.log` / `output.log` / `best_params.json` / `model.pkl` を
  ファイル名ベースで登録済み（どの階層にあっても自動的に無視される）。新しい種類の量産ファイル
  （例: 新モデルが独自の拡張子で結果を吐く場合）が出てきたら、同様にファイル名ベースで `.gitignore` に追記する。
- 一方、1実行あたり1ファイルの集計結果（`execution_times.csv`, `results.csv`, `analysis.json` 等、
  84データセットの結果を1枚にまとめたもの）はファイル数が少ないため追跡してよい。
- `git add -A` や `git add .` で量産ファイルが紛れ込んでいないか、コミット前に `git status` の
  追加/削除件数を確認する。`git add -f` で個別に無視ルールを上書きしない（過去に誤ってこれが行われ、
  `experiments/iiai_aai2026` と `experiments/undergraduate` 配下に84データセット分の `model.h5`/`training.log`
  が約1900ファイル誤って追跡されていたため、2026-11-04に`git rm --cached`で解消した）。
