# 距離・偏光角の尤度周辺化：現行手法のノート

対象: `dist_psi_marg/dpmarg`（2026-10-17 時点、反復 009 まで）。
導出の詳細は `dist_psi_marg.tex`（問題設定）、`TABLE1D.md`（距離テーブル）、`METHOD3.md`（Laplace 近似）、`REPORT.md`（全実験結果）を参照。
本ノートは「いま何をどう計算しているか」を一か所にまとめたもの。

---

## 1. 問題と記法

### 1.1 周辺化する量

$k$ 番目の検出器の対数尤度比を足し合わせると（`dist_psi_marg.tex` Eq. 17）

$$
\ln\mathcal L(D_L,\psi)=\frac{D_{\rm ref}}{D_L}\,(A\cos2\psi+B\sin2\psi)
-\frac12\Big(\frac{D_{\rm ref}}{D_L}\Big)^2\,(C\cos4\psi+D\sin4\psi+E),
\qquad E\ge\sqrt{C^2+D^2}.
$$

事前分布 $p(D_L)\propto D_L^2$（$D_{\min}\le D_L\le D_{\max}$）、$\psi$ 一様（$[0,\pi)$）で周辺化した尤度比

$$
\mathcal L^{\rm marg}=\frac{3}{\pi\,(D_{\max}^3-D_{\min}^3)}\int_{D_{\min}}^{D_{\max}}D_L^2\,dD_L\int_0^{\pi}d\psi\,\mathcal L
$$

を、$r=D_{\rm ref}/D_L\in[x_{\min},x_{\max}]$（$x_{\min}=D_{\rm ref}/D_{\max}$, $x_{\max}=D_{\rm ref}/D_{\min}$）、$\theta=2\psi\in[0,2\pi)$ で書き直すと

$$
\boxed{\;
\ln\mathcal L^{\rm marg}
=\ln\frac{3}{2\pi\,(x_{\min}^{-3}-x_{\max}^{-3})}
+\ln\int_{x_{\min}}^{x_{\max}}r^{-4}\,dr\int_0^{2\pi}d\theta\;
e^{\,r(A\cos\theta+B\sin\theta)-\frac12 r^2(C\cos2\theta+D\sin2\theta+E)}
\;}
$$

$D_{\rm ref}$ は陽に現れず、無信号（$A=\dots=E=0$）で $\ln\mathcal L^{\rm marg}=0$ となる。
スケール変換 $(A,B,C,D,E,x_{\min},x_{\max})\to(A/c,B/c,C/c^2,D/c^2,E/c^2,c\,x_{\min},c\,x_{\max})$ は値を変えないので、
テーブル類は比 $x_{\max}/x_{\min}=D_{\max}/D_{\min}$ ごとに 1 つ用意すれば十分（§7）。

### 1.2 縮約パラメータ

$$
R=\sqrt{A^2+B^2},\quad \zeta={\rm atan2}(B,A),\quad
S=\sqrt{C^2+D^2},\quad \xi=\tfrac12{\rm atan2}(D,C),\quad
\lambda_1=E+S\;\ge\;\lambda_2=E-S\;\ge0,\quad \eta=\zeta-\xi .
$$

$\theta\to\theta+\xi$ と回転すると

$$
\ln\mathcal L=a(\theta)\,r-\tfrac12 b(\theta)\,r^2,\qquad
a(\theta)=R\cos(\theta-\eta),\qquad
b(\theta)=\lambda_2+(\lambda_1-\lambda_2)\cos^2\theta=E+S\cos2\theta .
$$

全周積分の対称性（$\theta\to\theta+\pi$ で $a\to-a$、$\theta\to-\theta$ で $\eta\to-\eta$）により $\eta\in[0,\pi/2]$ に畳める。
物理的には $\lambda_2=0$ なら $\eta=0$（tex の議論）だが、コードは任意の $\lambda_2\ge0,\ \eta$ を受け付ける。

SNR の定義: $\rho^2=2\max\big(0,\max_{r\in[x_{\min},x_{\max}],\theta}\ln\mathcal L\big)$。

### 1.3 既定の設定

$(x_{\min},x_{\max})=(0.1,100)$（$D_{\max}/D_{\min}=1000$）を主、$(0.1,10)$ を副とする。精度目標は $|\Delta\ln\mathcal L^{\rm marg}|\le10^{-3}$（実測は全集合で $\le10^{-4}$）。

---

## 2. 計算の分解：距離 1 次元カーネル × 偏光角積分

$\theta$ を固定すると距離積分は distmarg.pdf と同じ 1 次元の形

$$
J(a,b)=\int_{x_{\min}}^{x_{\max}}r^{-4}\,e^{ar-br^2/2}\,dr,\qquad
f_{\max}(a,b)=\max_{r}\big(ar-\tfrac12 br^2\big),\qquad
\ln I(a,b)=\ln J-f_{\max}
$$

（$f_{\max}$ は $x_0=a/b$ を $[x_{\min},x_{\max}]$ にクリップした点の値；$b=0$ なら端点）。
$\theta\to\theta+\pi$ で $a\to-a$ なので、$\pi$ 周期の「対」

$$
h(\theta)=J(a(\theta),b(\theta))+J(-a(\theta),b(\theta)),\qquad
\ln\mathcal L^{\rm marg}=\ln\frac{3}{2\pi(x_{\min}^{-3}-x_{\max}^{-3})}+\ln\int_0^{\pi}h(\theta)\,d\theta
$$

を半周期で積分する。$J(-|a|)$ は $f_{\max}$ の差が $-40$ 未満で、かつ $2|a|x_{\min}>40$ または積分上界による保証が成り立つときは省略する（寄与 $<e^{-40}$）。

したがって手法は「**距離方向は $(a,b)$ の 2 変数テーブル**、**偏光角方向は $\theta$ の 1 次元数値積分**」である。
4 次元テーブル（手法 1）は不可、Laplace 近似のみ（手法 3）は高 SNR 限定、という比較結果は `REPORT.md` §3 参照。

---

## 3. 距離カーネル $\ln I(a,b)$

### 3.1 直接計算（`dpmarg/kernel1d.py`、検証・フォールバック用）

ピーク $r_*$（クリップした $x_0$）を中心とする対数座標 $v=\ln(x/r_*)$ で $\ln I=-3\ln r_*+\ln\int e^{-3v+f(r_*e^v)-f_{\max}}\,dv$ を
24 点 Gauss–Legendre の複合パネルで積分する。$v$ 座標の初期パネル幅は $\ln\!\big(1+\min(r_*/4,\,1/\sqrt b,\,1/|f'(r_*)|)/r_*\big)$ で、2 倍ずつ 0.5 まで広げ、
指数の低下が $H=1-\ln(10^{-12})+4\ln(x_{\max}/x_{\min})$ に達したところで打ち切る（凸性から相対打ち切り誤差 $\le10^{-12}$）。
精度 $\le10^{-9}$、1 回 3〜7 µs。

### 3.2 2 次元テーブル（`dpmarg/table1d.py`、本番）

**座標**（distmarg.pdf の $s,t$ 座標を $b\to0$ まで連続化したもの）

$$
s=\mathrm{asinh}\frac{a-bx_{\min}}{\sigma}+\mathrm{asinh}\frac{a-bx_{\max}}{\sigma},\qquad
v=\mathrm{asinh}\frac{b}{b_{\rm ref}},\qquad
\sigma=\sqrt{b+\sigma_0^2},\quad
\sigma_0=\frac1{x_{\max}-x_{\min}},\quad b_{\rm ref}=\frac1{x_{\max}^2}.
$$

$a-bx_{\min}$, $a-bx_{\max}$ は両端での指数の傾き（$=f'(x_{\min}),f'(x_{\max})$）で、ピークが端を横切る遷移を幅 $1/\sqrt b$ で引き伸ばす。$(s,v)\to(a,b)$ の逆写像は閉形式。

**格納量**: $\ln I$ そのものではなく、事前分布 $r^{-4}$ を含めない解析的な切断ガウス積分

$$
G(a,b)=\int_{x_{\min}}^{x_{\max}}e^{f(x)-f_{\max}}\,dx
=\sqrt{\frac{\pi}{2b}}\Big[\mathrm{erf}\big(\sqrt{b/2}\,(x_{\max}-x_0)\big)-\mathrm{erf}\big(\sqrt{b/2}\,(x_{\min}-x_0)\big)\Big]e^{\,bx_0^2/2-f_{\max}}
$$

（端点側は erfcx；端点傾き $d$ と $L=x_{\max}-x_{\min}$ に対し $dL+bL^2/2<10^{-3}$ なら Taylor 展開、$b=0$ は expm1 で安定化）との差 $H=\ln I-\ln G$ を格納する。
$H$ は $[-4\ln x_{\max},\,-4\ln x_{\min}]$ に収まる滑らかな量で、参照時に $\ln G$ を足し戻す。
事前分布をピークで凍結した残差 $\ln I-\ln(r_*^{-4}G)$ は $a=b=0$ で $4\ln(x_{\max}/x_{\min})$ だけ跳ぶため使わない。

**格子と補間**: $996\times996$ 点、$|s|\le24$、$0\le b\le10^8$、3 次 B スプライン（係数を事前計算、16 点参照）。係数 $10^6$ 個 = 8 MB、構築 18〜30 秒（主事前分布 20〜29 s、副 18〜22 s；JIT を除きガード計算を含む）。

**ガードとフォールバック**
- 各セルを中心と 4 辺中点の 5 点で直接計算と照合し、誤差 $>2\times10^{-6}$ のセルにフラグ（主事前分布で 0.8%、副で 0.3%）。フラグ付きセルは直接計算（本番の θ 積分では、$(x_{\min},x_{\max})=(0.1,10),(0.1,100)$ かつ $f_{\max}\le100$, $|a|x_{\min}\le100$, $bx_{\min}^2\le100$ のときに限り、対数半径のパネル幅 $\le0.5$ の複合 GL12 固定則で高速化。直接計算との差 $9\times10^{-12}$）。
- $|s|>24$: $d=|f'(\text{端点})|$ として $p=1/(d\,r)\le10^{-3}$, $q=b/d^2\le10^{-3}$, $d(x_{\max}-x_{\min})>80+4\ln(x_{\max}/x_{\min})$ なら 18 項の端点層漸近級数 $\ln I\simeq-4\ln r-\ln d+\ln\sum_n T_n$、それ以外は直接計算。
- $b>10^8$: 直接計算。

**精度・コスト**: 実際の θ 節点 $10^5$ 点と座標空間の $10^5$ 点で最大誤差 $2\times10^{-6}$、参照 0.14〜0.24 µs（フォールバック込みのシャッフル参照；温まった θ 節点ループ内では 0.15 µs/評価）、フォールバック率 0.7〜1.5%。

---

## 4. 偏光角積分（`dpmarg/geometry.py`, `dpmarg/method2.py`, `rule='analytic'`）

### 4.1 代理関数と解析的なピーク予測

距離方向を Laplace 近似すると、$\ln h(\theta)$ は（内部ピークの場合）

$$
p(\theta)\simeq f_{\max}(\theta)-4\ln x_0(\theta)+\tfrac12\ln\frac{2\pi}{b(\theta)},\qquad
f_{\max}=\frac{R^2\cos^2(\theta-\eta)}{2\,b(\theta)},\quad x_0=\frac{a}{b},
$$

端点にクリップされる場合は $f_{\max}=a\,x_b-\tfrac12 bx_b^2$、$-4\ln x_b$、幅項は $-\ln|a-bx_b|$ となる。
幅項の使用には妥当性ガード（内部: 端まで $4/\sqrt b$ 以上かつ $r_*\sqrt b>6$；端点: $|a-bx_b|>6\sqrt b$ かつ $|a-bx_b|\,x_b>20$）を課し、ガード外では幅項なしの連続代理 $q=f_{\max}-4\ln r_*$ を使う。

**内部ピークの極大は閉形式**。$\varphi=2\theta$ とおくと $f_{\max}\propto(1+\cos(\varphi-2\eta))/(E+S\cos\varphi)$ で、停留条件は

$$
E\sin(\varphi-2\eta)-S\sin\varphi=S\sin2\eta
\;\Longleftrightarrow\;
\rho'\sin(\varphi-\delta)=S\sin2\eta,\quad
\rho'=\sqrt{(E\cos2\eta-S)^2+E^2\sin^22\eta},\;
\delta={\rm atan2}(E\sin2\eta,\,E\cos2\eta-S),
$$

極大は $\varphi^*=\delta+\arcsin(S\sin2\eta/\rho')$。数値的に安定な同値形は

$$
\boxed{\;\theta^*={\rm atan2}(\lambda_1\sin\eta,\;\lambda_2\cos\eta)\pmod\pi\;}
$$

（完成平方した Cartesian ピーク $(\bar s,\bar t)=(R\cos\eta/\lambda_1,\,R\sin\eta/\lambda_2)$ の方向）。$R$ に依存せず、事前分布・幅項による真の極大からのずれは $O(1/\rho^2)$（幅 $1/\rho$ に対して小さい）。Newton 1〜数歩で吸収する。

**候補集合と精密化**（走査なし）: $\theta^*$、$\eta$、$0$、$\pi/2$（弱軸の島）、$\eta+\pi/2\pm0.01$、ピークが $x_{\min},x_{\max}$ を横切る角 $\arccos\!\big(R\cos\eta/((\lambda_1-\lambda_2)x_b)\big)$ と $\pi-\arccos(\cdots)$、弱軸近傍で Newton 精密化した横断点（両符号・両端）、を種として、代理関数の解析的 1・2 階微分による安全化 Newton 法（最大 10 歩、$x_{\min}$ 横断は直接解く）で精密化。最良から 40 log 以内の候補を保持し、幅より近いものは併合。
高 SNR で内部ピークが支配的なとき（端まで $8/\sqrt b$ 以上かつ $q>60-4\ln x_{\min}$、または $\lambda_2/\lambda_1>0.01$ で端まで $4/\sqrt b$ 以上かつ $p+\ln w>-4\ln x_{\min}+\ln\pi+25$）は候補探索を省き $\theta^*$ のみを使う。縮退コーナー（$\lambda_2/\lambda_1<0.01,\ \eta<0.1$）では $\theta^*$ の Laplace 精密化点と、$\lambda_2/\lambda_1<0.01$ かつ $R/\lambda_1>0.05x_{\max}$ のときは $x_{\max}$ 横断の「肩」（質量が最良から 14 log 以内）も候補に加える。

**幅**
- 滑らかな極大: $w=\min\big(0.3,\,1/\sqrt{-p''(\theta^*)}\big)$（$p''$ は解析式）。
- クリップ遷移の尖点極大（$x_0(\theta)$ が $x_{\min}$ や $x_{\max}$ を横切る点）: 距離方向の遷移幅 $1/\sqrt b$ を θ に写した

$$
w_{\rm cusp}=\min\Big(\text{片側曲率幅},\;\frac{\sqrt b}{|a'-b'x_0|}\Big)=\min\Big(\cdots,\;\frac{1/\sqrt b}{|dx_0/d\theta|}\Big)
$$

  を用い、「尖点（非ガウス）」とフラグする（例: $\lambda_2=0,\eta=0$ の平坦な尾根では $w_{\rm cusp}=\sqrt{\lambda_1}/(R\tan\theta_c)$）。
- $a=b=0$（$\lambda_2=0,\eta=0$ の $\theta=\pi/2$）: 対の値は厳密に $2\int r^{-4}dr$ で有限。幅項が特異になるため $R^2>\lambda_1$ では候補から外し、$R^2\le\lambda_1$ では $\pi/2$ を厳密な有限曲率 $(\lambda_1-R^2)\langle r^2\rangle_{\rm prior}$ の唯一の候補とする。
- 非コーナーで尖点かつ $r_*\sqrt b<6$（孤立した径方向サドルとみなせない）場合は幅を片側曲率幅の大きい方に置き換え、グリッド点数を 64 以上にする（prior layer）。
- 候補は推定質量 $\approx f_{\max}-4\ln r_*+\ln(\text{径方向の層幅})+\ln w$ で順位付けし、支配的な候補を中心に取る。幾何では高さ 40 log 以内を保持するが、GH 混合と多区間 sinh では質量が最大から 18 log 以内の候補に絞る。16 点の裾プローブは GH の採否判定にのみ使う。

### 4.2 求積則の選択

予測した幅 $w$（と形状）から、**ガウス型求積**か**固定グリッド**かを決める。閾値 $w_c=0.2$。

| 分岐 | 条件 | 則 |
|---|---|---|
| GH（mode 3） | 保持候補がすべて $w<0.2$ で幅項ガードを満たし、$\pm6w$ の外の裾の質量（16 点プローブ）が $10^{-6}\sqrt{2\pi}\,w\,e^{p^*}$ 未満、ピーク間距離 $>6(w_1+w_2)$、節点が 1 周期に収まる | Gauss–Hermite 12 点（縮退コーナー $\lambda_2/\lambda_1<0.01,\ \eta<0.1$ では 32 点、弱軸から $20w$ 以上離れていること） |
| 固定グリッド（mode 4） | $w_{\min}\ge0.2$；または GH 条件を満たさず（尖点を含む）、安全係数を掛けた幅の最小値が $\ge0.017$ | 等間隔周期グリッド $\theta_j=j\pi/N$、重み $\pi/N$、$N\in\{48,64,96,128\}$ の最小で $\pi/N\le1.3\,w$。GH 却下後の経路では $w$ に安全係数 0.8（コーナーでは 0.25）を掛け、無軸近傍（$a x_{\max}<4$, $bx_{\max}^2<4$）や質量の大きい弱軸光線ではさらに幅を縮める。prior layer では $N\ge64$ |
| sinh（mode 5） | GH 条件を満たさず、縮小後の最小幅が $<0.017$ | 単一の鋭いピークなら全周期の 64 点 sinh 伸張台形則；尖点・複数の鋭いピーク・鋭いピーク+他候補なら候補の中点で区切った多区間 sinh（各区間 64 点、区間幅 $1.5w$、コーナーの単一区間は $2w$、幅比 $>10$ なら $w^{1/4}$ 重みで分割、8 次 Euler–Maclaurin の片側端点補正） |
| 分割（mode 6） | コーナー外・尖点なしで、幅比 10 未満の 2 つ以上の鋭いピークが重なる | 隣接候補の中点で区間分割し、各区間で幅 $1.5w$（上限 0.3）の sinh 写像 + GL32 |

**Gauss–Hermite**: $\theta_j=\theta^*+\sqrt2\,w\,t_j$（$t_j,\omega_j$ は Hermite 節点・重み、import 時に事前計算）

$$
\int h\,d\theta\;\approx\;\sqrt2\,w\sum_j\omega_j\,e^{t_j^2}\,h(\theta_j)
$$

を log 空間で累積する。GH は尖点には使わない。

**固定グリッド**（Roulet et al. 2024, arXiv:2404.02435 §II.3 の等間隔位相グリッド + 台形則に倣う）: $\cos\theta_j,\sin\theta_j$ を事前計算し、
$a_j=R(\cos\theta_j\cos\eta+\sin\theta_j\sin\eta)$、$b_j=\lambda_2+(\lambda_1-\lambda_2)\cos^2\theta_j$ を超越関数なしで得る。周期的で滑らかな被積分関数に対し台形則は指数収束し、幅 $w$ のガウス状ピークの誤差は $\sim2e^{-2\pi^2(w/h)^2}$（$h=\pi/N$）。

**sinh 伸張台形則**（単一区間）: $\theta=\theta^*+w\sinh\tau$、$\tau$ を $[-T,T]$（$w\sinh T=\pi/2$）に等間隔 64 点、重み $d\tau\,w\cosh\tau$。周期性から第 1 Euler–Maclaurin 端点補正は $-\pi\,d\tau^2\,h(\text{端})/12$ と厳密に定まる。多区間版は非周期なので 8 次 Euler–Maclaurin の片側ステンシルで端点補正し、上下界判定で補正が信頼できない場合は無効化する。

### 4.3 寄与が無視できる節点のスキップ

各節点で代理関数 + $\ln(\text{重み})$ のスコアを計算し、最大から 40 を超えて下回り、かつ厳密な上界
$\ln h\le f_{\max}+\ln(2G_{\max})-4\ln x_{\min}$（$G_{\max}$ は指数部のみの積分の上界）に重みを足したものが、保持される最大節点の下界 $-25-\ln N$ を下回るときだけ表参照を省く（保持される最大項に対する相対損失 $\le e^{-25}$ を保証）。全節点の約 4% を省き、結果の変化は $4\times10^{-15}$ 以下。

### 4.4 1 呼び出しの流れ

1. 係数の縮約 $(\lambda_1,\lambda_2,R,\eta)$。
2. 解析的幾何（候補 → Newton → 幅・質量）: 中央値 約 2.1 µs（旧走査版 5.2 µs）。
3. 分岐選択（表 4.2）と節点・重みの生成。
4. 各節点で $(a_j,b_j)$ → 対の表参照 $\ln h(\theta_j)$（スキップ判定付き）。
5. log-sum-exp と正規化 $\ln\frac{3}{2\pi(x_{\min}^{-3}-x_{\max}^{-3})}$。全零係数は厳密に 0。

典型コスト（共有 4 集合の集計）: GH12 分岐 中央値 5.1 µs（10〜90 パーセンタイル 4.4〜15 µs）、固定グリッド分岐 $N=48$ で中央値 14 µs、$N=64$ で約 40 µs（グリッド全体の中央値 36 µs）、全体の中央値 6.7 µs。

---

## 5. 高 SNR の閉形式分岐（任意、`dpmarg/fast.py`）

内部ピークで両方向がガウス的なら、Cartesian 主軸 $(s,t)$ での 2 次元 Laplace 展開（`METHOD3.md`）

$$
\ln\mathcal L^{\rm marg}\simeq{\rm norm}+f_{\rm peak}+\ln2\pi-\tfrac12\ln(\lambda_1\lambda_2)-5\ln\bar r+\ln(1+C_2+C_4),
$$

$C_2=\tfrac12[(-5+35v)z_1+(-5+35w)z_2]$、$z_i=1/(\lambda_i\bar r^2)$、$v=\bar s^2/\bar r^2$、$w=\bar t^2/\bar r^2$（$C_4$ は 4 次項）。
`lnL_marg_fast` はゲート（$\lambda_2/\lambda_1\ge10^{-3}$、$3/\sqrt{\lambda_2}$ の楕円が環内、$\rho\ge1.5\,\rho_{\rm break}(q)$、$\rho_{\rm break}$ は $(q,\rho)=(10^{-3},422),(10^{-2},133),(0.1,42),(0.3,18)$ の対数補間）を通る場合だけこの式（0.15 µs）を使い、それ以外は §4 の手法 2 を呼ぶ。
呼び出しの 10〜23% が該当する。反復 008 の計測（旧走査則）では平均コストは下がるが中央値は改善しなかった（手法 2 単独 20.3 µs vs ディスパッチャ 21.2 µs）；解析則が既定になった後のディスパッチャ単体の時間計測は未実施（精度は最大 3.3e-4 を確認済み）。`use_laplace=False` で手法 2 単独になる。

---

## 6. 精度とコスト（2026-10-17、単一スレッド、JIT 後）

| 集合 | 点数 | 最大誤差（解析 / 旧走査） | 中央値 / 平均 / p99 µs（解析） | 同（旧走査） |
|---|---|---|---|---|
| 回帰（名前付き極端例 + 乱数） | 40 | 3.4e-5 / 3.4e-5 | 12.6 / 16.3 / 60 | 20.6 / 31.2 / 147 |
| 保存乱数 | 300 | 5.0e-7 / 4.7e-5 | 13.9 / 18.6 / 53 | 20.4 / 23.6 / 73 |
| 新規乱数（seed 808） | 2000 | 9.4e-5 / 9.4e-5 | 13.0 / 18.7 / 79 | 20.9 / 27.1 / 162 |
| 模擬事後分布（SNR 5–50、内部ピーク） | 2000 | 1.6e-5 / 3.3e-6 | 5.0 / 17.2 / 85 | 21.4 / 38.9 / 172 |
| 縮退コーナー格子（$\lambda_2/\lambda_1\le10^{-2}$, $\eta\le0.1$） | 300 | 7.5e-5 / 1.3e-4 | 25 / 32 / 123 | 31 / 35 / 75 |
| 縮退コーナー乱数 | 300 | 7.9e-5 / 3.8e-4 | 27 / 38 / 139 | 35 / 42 / 98 |

全体の中央値 6.7 µs。分岐の割合（共有 4 集合）: GH 51〜75%、グリッド 25〜47%、sinh/分割 $\le2.5$%（縮退コーナーでは GH 33〜54%、グリッド 9〜14%、sinh 31〜58%）。参照実装（scipy 入れ子適応求積と mpmath 25 桁、回帰 40 点で互いに $3.4\times10^{-11}$ で一致）は 0.1〜0.3 s/点で、検証専用。pytest 507 件。

---

## 7. 使い方

```python
from dpmarg import build_table, load_table, lnL_marg_fast, lnL_marg_m2_table
from dpmarg.table1d import save_table

table = build_table(0.1, 100.0)            # 約 20–30 秒、8 MB
save_table("data/table1d_xmin0.1_xmax100.npz", table)
table = load_table("data/table1d_xmin0.1_xmax100.npz")

value = lnL_marg_m2_table(A, B, C, D, E, 0.1, 100.0, table)   # 手法 2（既定: rule='analytic'）
value = lnL_marg_fast(A, B, C, D, E, 0.1, 100.0, table)       # 高 SNR 閉形式分岐付き
```

- シグネチャは `lnL_marg_m2_table(A,B,C,D,E,xmin,xmax,table,n_nodes=64,rule='analytic')`。`n_nodes` は `rule='scan'` のときの最大節点予算で、`'analytic'` では無視される。
- 配列版 `dpmarg.method2.lnL_marg_m2_table_array(parameters, table, n_nodes=64, rule='analytic')` / `dpmarg.fast.lnL_marg_fast_array(parameters, table, use_laplace=True)` は $(n,7)$ 行列 $(A,B,C,D,E,x_{\min},x_{\max})$ を受け取る（トップレベル `dpmarg` からは import されないので各モジュールから import する）。
- 他の事前分布範囲: 比 $x_{\max}/x_{\min}$ が同じなら $c$ でテーブルの範囲に写し、$A,B$ を $1/c$、$C,D,E$ を $1/c^2$ 倍して評価する（API は自動では行わない）。比が異なればテーブルを作り直す。
- 診断: `dpmarg.method2.method2_analytic_diagnostics(A,B,C,D,E,xmin,xmax,table, gh_order=12, grid_n=0)` が（値, 分岐 3/4/5/6, 総節点数, 最小幅, 半径カーネル評価回数（表参照 + フォールバック、節点あたり 1〜2 回）, フォールバック数, スキップ数）の 7 要素タプルを返す。旧ルールは `rule='scan'`（ただし縮退コーナー $\lambda_2/\lambda_1<0.01,\ \eta<0.1$ で `n_nodes=64` のときは解析則の固定求積に切り替わる）。

---

## 8. 既知の限界

- 精度は文書化した集合（約 5,000 点 + 縮退コーナー 600 点）での実測であり、全係数に対する一様な誤差保証ではない。ガードセルのプローブも 5 点の経験則。
- 縮退コーナー（$\lambda_2/\lambda_1<10^{-2}$, $\eta<0.1$）は安全側の設定（GH32、$N$ 最大 128）でコストが 2〜4 倍。
- 事前分布は $D_L^2$ 専用（$r^{-4}$ の重みと正規化がカーネルに組み込まれている）。
- 分岐の切り替えは不連続で、勾配を使うサンプラー向けの滑らかさは保証しない。
- 距離テーブルは事前分布の比ごとに構築が必要（$b\le10^8$、$|s|\le24$ の外は直接計算に落ちる）。
