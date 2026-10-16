# TASK.md — 距離・偏光角に関する尤度周辺化の高速化

（オーケストレーター案。プロジェクトオーナーの確認を経て確定する。）

## 1. 目的

重力波パラメータ推定において、尤度比 $\mathcal{L}$ を光度距離 $D_L$ と偏光角 $\psi$ について
周辺化した
$$
\mathcal{L}^{\rm marg} = \frac{3}{\pi (D_{\max}^3 - D_{\min}^3)} \int_{D_{\min}}^{D_{\max}} D_L^2\, dD_L \int_0^{\pi} d\psi\, \mathcal{L},
\qquad
\ln \mathcal{L} = \frac{D_{\rm ref}}{D_L}(A\cos 2\psi + B \sin 2\psi) - \frac{1}{2}\left(\frac{D_{\rm ref}}{D_L}\right)^2 (C \cos 4\psi + D \sin 4\psi + E)
$$
（`dist_psi_marg.tex` Eq. (17)）を、サンプリング中に毎回（$10^6$–$10^8$ 回）呼べる速さ
（目標: numba 実装で 1 呼び出しあたり数 µs〜数十 µs）と精度（目標: $\ln \mathcal{L}^{\rm marg}$ の
絶対誤差 $< 10^{-3}$、少なくとも $< 10^{-2}$）で計算する手法を、数値実験に基づいて決める。

対象とする係数 $(A, B, C, D, E)$ は、重力波で典型的な値だけでなく、サンプリング中に現れうる
あらゆる値（無信号 $R=0$、縮退 $\lambda_2 = 0$、極端に大きい $\lambda_1$、ピークが
$[D_{\min}, D_{\max}]$ の外、$\psi$ 方向に二峰的、SNR 0〜100 超）を含む。

## 2. 数学的整理（オーケストレーターの導出。Codex は検証して用いる）

$x = (D_{\rm ref}/D_L)\cos 2\psi$, $y = (D_{\rm ref}/D_L)\sin 2\psi$ とし、
$A = R\cos\zeta$, $B = R \sin\zeta$, $C = S\cos 2\xi$, $D = S\sin 2\xi$ とおいて $(x,y)$ を
$\xi$ だけ回転した座標 $(s, t)$ では
$$
\ln\mathcal{L} = -\tfrac{\lambda_1}{2} s^2 - \tfrac{\lambda_2}{2} t^2 + R\cos\eta\, s + R\sin\eta\, t,
\qquad \lambda_1 = E + S \ge \lambda_2 = E - S \ge 0,\quad \eta = \zeta - \xi .
$$
事前分布は極座標 $r = D_{\rm ref}/D_L \in [x_{\min}, x_{\max}] = [D_{\rm ref}/D_{\max}, D_{\rm ref}/D_{\min}]$、
$\theta = 2\psi \in [0, 2\pi)$ で $\propto r^{-4}\, dr\, d\theta$（$(x,y)$ 平面では $r^{-5} dx\,dy$）。
$\theta$ の積分が全周にわたるため回転は対称性であり、周辺化尤度は固定された
$(x_{\min}, x_{\max})$ のもとで **4 変数 $(\lambda_1, \lambda_2, R, \eta)$ の関数**になる
（鏡映対称性により $\eta \in [0, \pi/2]$ に制限できる）。
$\theta$ を固定すると被積分関数は距離のみの周辺化（`distmarg.pdf`）とまったく同じ形
$a(\theta)\, r - \tfrac{1}{2} b(\theta) r^2$, $a = R\cos(\theta-\eta)$,
$b = \lambda_1 \cos^2\theta + \lambda_2 \sin^2\theta$ になる。

## 3. 検討する 3 手法

1. **全変数の Lookup table**: 対称性で 4 次元に落とした空間に、`distmarg.pdf` の
   $\sinh^{-1}$ 座標を一般化した格子を張り、多重線形または 3 次補間で済ませる。
2. **部分 Lookup table + 毎回の数値積分**: 例えば既存の 2 次元距離テーブル
   $\ln I(x_0, b)$ を $\psi$ の節点ごとに参照し、$\psi$ については少数（16〜64 点）の
   適応的な節点で積分する（他の変数分割も比較する）。
3. **急峻方向のガウス近似 + 残りの Lookup table**（`dist_psi_marg.tex` Sec. 2–3）。
   高 SNR でのみ成立するため、破綻する SNR を定量化し、フォールバックを決める。

## 4. 完了判定

（統合分析の結果を反映して確定する。下記は案。）

- `dpmarg.reference`: 全レジームで $\ln\mathcal{L}^{\rm marg}$ を絶対誤差 $\lesssim 10^{-8}$ で与える
  総当たり参照実装と、代表点・極端点の回帰リストがあり、独立な 2 法で一致を確認済み。
- 3 手法それぞれについて、同じ参照実装・同じパラメータ集合に対する
  精度（絶対誤差の分布・最悪値）、コスト（1 呼び出しあたりの中央値 µs、テーブル構築時間、
  メモリ）、頑健性（極端レジームの合否）を数値実験で測り、図と表にまとめてある。
- 最良手法（または組合せ）の実装が `pytest` で検証され（参照との差 $<10^{-3}$ を既定、
  許容を緩める場合はその理由を明記）、ベンチマークが目標速度を満たす。
- `dist_psi_marg/REPORT.md` に、結論（どの手法を推すか、どの SNR 以下でガウス近似が
  破綻するか、テーブル格子の設計）と図が記されている。

## 5. 制約と編集禁止

- 作業は `dist_psi_marg/` 配下のみ。`dist_psi_marg.tex`, `dist_psi_marg.pdf`, `distmarg.pdf`
  は編集禁止（読み取り専用の仕様書）。リポジトリ外・ `.devcontainer/` ・`main` は触らない。
- Codex は commit / push / stash / reset を行わない。環境へのパッケージ追加も行わず、
  必要なら報告して終了する（オーケストレーターが `environment.yml` と環境を更新する）。
- 秘密情報をファイル・ログ・コミットに残さない。
- 図は `dist_psi_marg/figures/`、データ・生成テーブルは `dist_psi_marg/data/`
  （大きな生成テーブルは git に入れない。`.gitignore` で管理）。

## 6. 環境

- クラウド: `bash dist_psi_marg/tools/cloud_setup.sh` で micromamba（conda-forge 由来）と
  `environment.yml` の conda 環境を `/home/user/conda-env` に作り、`PATH` の先頭に加える。
- ローカル: `/workspace/.conda-linux`（`which python` で確認）。
- Codex 呼び出し: `sh .devcontainer/run_codex.sh exec --sandbox workspace-write ...`
  （このクラウドコンテナではサンドボックスの動作を確認済み）。

## 7. 反復計画

（統合分析の結果を反映して確定する。）

## 8. 未決事項（オーナーの判断が必要）

（統合分析の結果を反映して確定する。）
