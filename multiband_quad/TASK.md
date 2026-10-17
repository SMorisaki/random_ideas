# TASK.md — 高次モードを含む波形の (h, h) を multibanding で速く計算する

（2026-10-17 草案。オーナーの合意を得てから反復を開始する。§8 の未決事項への回答で更新する。）

## 1. 目的

Morisaki (2021), PRD 104, 044062, arXiv:2104.07813（`notes/arXiv_2104.07813v2.pdf`、要約
`notes/paper_summary.md`）の multibanding では、尤度の 2 次項
$$(h,h) = \frac{4}{T}\sum_k \frac{|\tilde h(f_k)|^2}{S_k}$$
を、高次モード (HOM) を含む波形について **IFFT-FFT 法**（Sec. II.D.2, Eqs. 31–46）で計算している。
これは各バンドの粗い格子上の波形を IFFT → ゼロ詰め → FFT で 2 倍細かい格子へ補間し、折り畳んだ
PSD の重みと掛ける方法で、精度は良いが尤度評価ごとにバンド数だけ IFFT と FFT を要し、
波形生成に比べて無視できない時間がかかる。

本タスクのゴールは、**同等以上の精度で、実行時に FFT を使わず、IFFT-FFT 法より大幅に速い (h,h) の
計算法を確立し、数値実験で実証すること**である。

## 2. 提案手法（オーケストレーターの導出。`notes/method_idea.md` を参照）

- 波形をモード和 $h=\sum_A h_A$（$A=(l,m)$、非歳差）に分け、$|h|^2 = \sum_A |h_A|^2 + 2\,\mathrm{Re}\sum_{A>A'} h_A^* h_{A'}$ とする。
- 各順序付きモード対の積 $w^{(b)} h_A^* h_{A'}$ の「時間（ラグ）方向の台」は、
  定常位相近似より $\mathrm{lag}(f) = \tau_A(f) - \tau_{A'}(f)$ の値域で、遅いモードを先にとれば
  **片側**（$[0, \mathrm{lag}_{\max}]$）かつ $\mathrm{lag}_{\max} \lesssim \tau_{m_{\max}} < T^{(b)}$。
  （$t_c$ はラグに現れないので、論文の Eq. 51 に含まれる $t_c$ 事前分布幅 2.12 s は不要。）
- よって各対は **(d,h) と同じ粗い格子**の上で、対ごとに「時間ブロック」をそのラグ区間に合わせた
  事前計算済みの重み
  $$W_\kappa(B) = \frac{1}{\hat N}\sum_{m\in B} A_m^*\, e^{2\pi i m\kappa/\hat N},\qquad A_m=\sum_k \frac{e^{2\pi i k m/N^{(b)}}}{S_k}$$
  を使って $\sum_k a_k g_k = \sum_\kappa W_\kappa(B)\, g(\kappa/\hat T)$ と厳密に（台がブロック内なら）計算できる。
  論文の Eq. 45 は「全波形 $|h|^2$ に対称ブロック $2T^{(b)}$ を使った特殊例」であり、対称ブロックが
  必要だったのは $|h|^2$ 全体のラグ台が $[-T^{(b)}, T^{(b)}]$ と両側に広がるため。
- 実行時コストは $O(n_{\rm modes}^2 K_{\rm MB})$ の複素積和（バッチ化した 2 次形式）だけで FFT は不要。
  事前計算は PSD・窓・ブロックだけに依存する。
- トイ実験（`notes/orchestrator_toy.py`、`notes/method_idea.md` §6）では、IFFT-FFT 法と同じ
  相対誤差 ~2–4e-7 を粗い格子だけで達成した（線形補間法は 1.6e-6、ブロックの置き方を誤ると 4.4e-6）。
  誤差は $\sum_{m\notin B} A_m^* G_m$ で決まるので、PSD が滑らかだとブロックの置き方に鈍感になり、
  **狭い線スペクトルがあるとき**に置き方が効く。試験には線を含む PSD が必須。

要件: 波形モデルが各モード（または各 co-precessing モード）を任意周波数列で返せること。
IMRPhenomXHM（`SimIMRPhenomXHMFrequencySequenceOneMode`、環境に導入済み）がこれを満たす。
IFFT-FFT 法は全波形 $h$ しか要らないので、その点は IFFT-FFT 法の利点として記録する。

## 3. 比較対象

1. 総当たり参照: 細かい格子での Eq. 4。
2. 線形補間法（Eqs. 27–30）。
3. IFFT-FFT 法（Eqs. 31–46）を論文どおりに再実装したもの（公平を期して numpy で合理的に最適化）。
4. 参考: 全波形を $2T^{(b)}$ 格子（半整数点）で直接評価して Eq. 45 の重みを使う方法
   （FFT もモード分解も不要だが波形評価が 2 倍）。

## 4. 完了判定（案。§8 の回答で確定）

- **C1 精度**: 合成 SPA 多モード波形と IMRPhenomXHM（非歳差、モード (2,±2),(2,±1),(3,±3),(3,±2),(4,±4)）について、
  (i) BNS 1.4–1.4 $M_\odot$（$f_{\rm low}=20$ Hz, $T=256$ s, $\{T^{(b)}\}=\{256,\dots,4\}$ s）、
  (ii) GW190814 型 23+2.6 $M_\odot$（$T=16$ s, $\{16,8,4\}$ s）、(iii) BBH 36+29 $M_\odot$（$T=8$ s, $\{8,4\}$ s）
  と、各クラス ≥100 点の乱数母集団（質量・整列スピン・傾斜角・位相・$t_c$ は事前分布内）に対し、
  $L=5$ と $L=50$ で、単一検出器 SNR 25（補助的に 50, 100）に規格化した $|\Delta\ln\Lambda| = |\Delta(h,h)|/2$ の
  中央値と 99 パーセンタイルが、同じ $L$ の IFFT-FFT 法の値以下（99 パーセンタイルは 2 倍以内まで許容）。
  PSD は aLIGO 設計感度 + 合成の狭い線スペクトル、および実データ由来の PSD（§8）。
- **C2 速度**: 粗い格子上のモード配列を入力としたときの (h,h) ステップの中央値時間が、IFFT-FFT 法の
  ステップの 1/5 以下、かつ IMRPhenomXHM のモード評価時間の 10% 以下（(ii) と (i) の設定、この計算機、numpy。
  numba 版も報告）。事前計算は (i) の設定で 60 s 以下・200 MB 以下。
- **C3 設計則**: ブロックの置き方（予測ラグ区間の中心に置く）と余裕の規則、および HOM 用のバンド決定則
  （Eq. 51 に追加の余裕が要るか）が、余裕対誤差の実験で定量的に裏付けられている。
- **C4 文書と試験**: `METHOD_NOTE.md`（導出・設計則・限界）、`REPORT.md`（Fig. 4 型の誤差分布、コスト内訳、
  余裕の研究、線スペクトル頑健性の図）、`pytest`（恒等式: 全ブロックで $W_\kappa=a_\kappa$、共役対称、
  対称 $2T$ ブロック＋全波形で IFFT-FFT と一致、回帰精度）。

## 5. 制約と編集禁止

- 作業は `multiband_quad/` 配下のみ。`notes/arXiv_2104.07813v2.pdf`、`notes/paper_summary.md`、
  `notes/method_idea.md`、`notes/orchestrator_toy.py` は Codex 編集禁止（仕様・参考実装）。
  リポジトリ外・`.devcontainer/`・`main`・`dist_psi_marg/` は触らない。
- Codex は commit / push / stash / reset を行わない。パッケージ追加も行わず、必要なら報告して終了する
  （オーケストレーターが `environment.yml` と環境を更新する）。
- 秘密情報をファイル・ログ・コミットに残さない。
- 図は `multiband_quad/figures/`、データは `multiband_quad/data/`（大きな配列は git に入れない。`.gitignore` 済み）。
- 論文の記法（$T^{(b)}, N^{(b)}, M^{(b)}, K_s, K_e, w^{(b)}, L, \tau, \tilde D, \tilde I, c_k$）を名前と docstring に使う。

## 6. 環境

- クラウド: `bash dist_psi_marg/tools/cloud_setup.sh` で `/home/user/conda-env` を作成し、`PATH` の先頭に加える。
  `python-lalsimulation` 6.2.0 を追加済み（`environment.yml` 更新済み）。
- ローカル: `/workspace/.conda-linux`（`which python` で確認）。`environment.yml` の同期で lalsimulation が入る。
- Codex 呼び出し: `sh .devcontainer/run_codex.sh exec --sandbox workspace-write -c 'model_reasoning_effort="high"' -C multiband_quad ...`
  （このクラウドコンテナでサンドボックス動作を確認済み: `~/.codex/sandbox_mode = workspace-write`）。
- `/projects/data_release` はクラウドではマウントされていない。実 PSD は §8 の方針に従う。
- 計算機: 4 コア、15 GB RAM。

## 7. 反復計画（各反復 30 分以内を目安。合格条件は該当する完了判定）

| 反復 | 内容 | 合格条件 |
|---|---|---|
| 001 | パッケージ骨格 `mbquad/`: PSD（解析 aLIGO + 合成線、ASD ファイル読込）、合成 SPA 多モード波形、バンド決定（Eqs. 47, 51, 53, 55, 56）、窓（Eq. 9）、粗い格子（$N^{(b)}, M^{(b)}, K_s, K_e$）、総当たり (h,h)、pytest | 窓の分割統一性、バンド単調性、BNS 20 Hz/256 s で $K_{\rm orig}/K_{\rm MB}$ が Table I（27, $m_{\max}=4$）の ±15% |
| 002 | 論文の 2 法（線形補間 Eqs. 28–30、IFFT-FFT Eqs. 31–46）の忠実な再実装と事前計算、計時枠組み | トイで §2 の数値（線形 ~1e-6、IFFT-FFT ~2e-7）を再現 |
| 003 | 提案法 `mbquad.pairwise`: ブロック重み $W(B)$ の事前計算、ラグ予測器、ブロック配置則（余裕パラメータ付き）、バッチ 2 次形式 | 恒等式試験、トイで IFFT-FFT と同等精度、余裕対誤差の図 |
| 004 | IMRPhenomXHM 接続: モード別の任意周波数列評価（±m）、検出器射影 $h = F_+ h_+ + F_\times h_\times$ をモード別に分解、モード和が `ChooseFDWaveformSequence` と一致 | 一致 1e-10、3 系で (h,h) の 3 法比較 |
| 005 | 母集団実験（3 クラス × ≥100 点、$L=5, 50$、SNR 25/50/100）: 線形 / IFFT-FFT / 提案法の誤差分布（Fig. 4 型）、線スペクトル付き PSD と実 PSD | C1 の判定に必要なデータと図 |
| 006 | HOM 用バンド決定則と余裕の研究: 空き時間（slack）対誤差、基準チャープ質量と真の質量の差の影響、Eq. 51 の修正案 | C3 |
| 007 | 性能: numpy einsum / numba 実装、IFFT-FFT 基準実装の公平な最適化、計時表（波形評価 vs 各法の (h,h) ステップ）、事前計算コストとメモリ | C2 |
| 008 | BNS 本番規模（$T=256$ s, 20–2048 Hz, $N=2^{20}$）での端から端までの検証 | C1, C2 を (i) で確認 |
| 009 | `METHOD_NOTE.md`、`REPORT.md`、図、最終 pytest、整理 | C4 |
| 010+ | 発展（オーナーの判断）: 歳差波形への拡張（IMRPhenomXPHM の co-precessing モード別評価）、bilby `MBGravitationalWaveTransient` への移植メモ | — |

## 8. 未決事項（オーナーの判断が必要）

以下はオーケストレーターの仮置き。回答があれば TASK.md を更新する。

1. 精度目標: 「IFFT-FFT（同じ $L$）以下、99 パーセンタイルは 2 倍以内」でよいか。SNR の規格化は単一検出器 SNR 25 を主とする。
2. 速度目標: 「(h,h) ステップが IFFT-FFT の 1/5 以下かつ波形評価の 10% 以下」でよいか。
3. 波形モデル: IMRPhenomXHM（非歳差）を主とする。論文の IMRPhenomHM も `ChooseFDModes` で追加可能。歳差は範囲外（発展項目）でよいか。
4. 実 PSD: クラウドから GWOSC / LIGO DCC / zenodo に到達できる（HTTP 200 確認済み）。GW190814 周辺の公開ストレインから Welch 推定するか、
   GWTC PE データリリースの PSD を使う案。ダウンロードしてよいか（`data/` に保存、git には入れない）。不可なら合成線スペクトルのみで行う。
5. 検出器応答は単一検出器・固定 $(F_+, F_\times)$ でよいか（ネットワークは各検出器の和なので本質は変わらない）。
6. bilby への移植は範囲外とし、bilby の `MBGravitationalWaveTransient` に移しやすい関数構成（事前計算オブジェクト＋評価関数）にとどめる。
