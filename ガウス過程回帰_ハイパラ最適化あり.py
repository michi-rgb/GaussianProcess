# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 15:21:30 2025

@author: user
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, log, sqrt
from numpy.linalg import det, inv
from scipy.optimize import minimize

# plot parameters
N = 100
xmin = -1
xmax = 3.5
ymin = -1
ymax = 3
blue = '#ccccff'

# ガウス過程回帰モデルのハイパーパラメータを最適化するための、対数尤度の最小化履歴
loglik_result_list = []

def kgauss(params):
    """
    ガウスカーネルを定義します。

    Args:
        params (list): [tau, sigma, eta]

    Returns:
        function: ガウスカーネル関数
    """
    [tau, sigma, eta] = params
    return lambda x, y, train=True: \
        exp(tau) * exp(-(x - y)**2 / exp(sigma)) + (exp(eta) if (train and x == y) else 0)

def kgauss_grad(xi, xj, d, kernel_func, params):
    """
    カーネル関数のパラメータに関する勾配を計算します。

    Args:
        xi, xj (float): データ点
        d (int): パラメータのインデックス (0:tau, 1:sigma, 2:eta)
        kernel_func (function): ガウスカーネル関数
        params (list): パラメータ

    Returns:
        float: 勾配値
    """
    K = kernel_func(params)(xi, xj)
    if d == 0:
        return exp(params[d]) * K
    if d == 1:
        return K * (xi - xj)**2 / exp(params[d])
    if d == 2:
        return (exp(params[d]) if xi == xj else 0)
    else:
        return 0

def kv(x, xtrain, kernel):
    """
    テストデータと訓練データの間のカーネル値のベクトルを計算します。

    Args:
        x (float): テストデータ点
        xtrain (np.ndarray): 訓練データ点
        kernel (function): カーネル関数

    Returns:
        np.ndarray: カーネル値のベクトル
    """
    return np.array([kernel(x, xi, False) for xi in xtrain])

def kernel_matrix(xx, kernel):
    """
    訓練データのカーネル行列を計算します。

    Args:
        xx (np.ndarray): 訓練データ点
        kernel (function): カーネル関数

    Returns:
        np.ndarray: カーネル行列
    """
    N_data = len(xx)
    return np.array(
        [kernel(xi, xj) for xi in xx for xj in xx]
    ).reshape(N_data, N_data)

def gpr(xx, xtrain, ytrain, kernel):
    """
    ガウス過程回帰を実行します。

    Args:
        xx (np.ndarray): 予測を行うテストデータ点
        xtrain (np.ndarray): 訓練データ点
        ytrain (np.ndarray): 訓練データの値
        kernel (function): カーネル関数

    Returns:
        tuple: 予測された平均値と分散のタプル
    """
    K = kernel_matrix(xtrain, kernel)
    Kinv = inv(K)

    # Kをヒートマップで可視化(ガウス過程回帰の実行時)
    plt.figure(figsize=(8, 6))
    plt.imshow(K, cmap='viridis', origin='lower')
    plt.colorbar(label='Kernel Value')
    plt.title('Kernel Matrix Heatmap')
    plt.xlabel('Training Data Index')
    plt.ylabel('Training Data Index')
    plt.show()

    ypr = []; spr = []
    for x in xx: # xxは長さNの等差数列(このテストデータの結果をプロット)
        s = kernel(x, x)
        k = kv(x, xtrain, kernel) # テストデータと訓練データの間のカーネル値のベクトルを計算

        ypr.append(k.T.dot(Kinv).dot(ytrain)) # 予測平均
        spr.append(s - k.T.dot(Kinv).dot(k)) # 予測分散

    return np.array(ypr), np.array(spr)

def loglik(params, xtrain, ytrain, kernel_func, kgrad_func):
    """
    対数尤度を計算します。

    Args:
        params (np.ndarray): パラメータ
        xtrain (np.ndarray): 訓練データ点
        ytrain (np.ndarray): 訓練データの値
        kernel_func (function): カーネル関数
        kgrad_func (function): 勾配関数

    Returns:
        float: 対数尤度
    """
    K = kernel_matrix(xtrain, kernel_func(params))
    Kinv = inv(K)
    loglik_result = log(det(K)) + ytrain.T.dot(Kinv).dot(ytrain)
    loglik_result_list.append(loglik_result) # 履歴出力用

    return loglik_result

def gradient(params, xtrain, ytrain, kernel_func, kgrad_func):
    """
    対数尤度の勾配を計算します。

    Args:
        params (np.ndarray): パラメータ
        xtrain (np.ndarray): 訓練データ点
        ytrain (np.ndarray): 訓練データの値
        kernel_func (function): カーネル関数
        kgrad_func (function): 勾配関数

    Returns:
        np.ndarray: 勾配ベクトル
    """
    K = kernel_matrix(xtrain, kernel_func(params))
    Kinv = inv(K)
    Kinvy = Kinv.dot(ytrain)
    D = len(params)
    N_data = len(xtrain)
    grad = np.zeros(D)

    # Kをヒートマップで可視化(ガウス過程回帰のハイパーパラメータ最適化時)
    plt.figure(figsize=(8, 6))
    plt.imshow(K, cmap='viridis', origin='lower')
    plt.colorbar(label='Kernel Value')
    plt.title('Kernel Matrix Heatmap')
    plt.xlabel('Training Data Index')
    plt.ylabel('Training Data Index')
    plt.show()

    for d in range(D):
        G = np.array(
            [kgrad_func(xi, xj, d, kernel_func, params)
             for xi in xtrain for xj in xtrain]
        ).reshape(N_data, N_data)

        grad[d] = np.trace(Kinv.dot(G)) - Kinvy.T.dot(G).dot(Kinvy)
    return grad

def optimize(xtrain, ytrain, kernel_func, kgrad_func, init):
    """
    対数尤度を最大化するためにパラメータを最適化します。

    res.message の意味:
        Current function value: 現在の目的関数の値です。この目的関数は、ガウス過程モデルの対数尤度を計算しています。
        最適化アルゴリズムは、この値が最も小さくなるようにパラメータを調整していきます。値が小さいほど、モデルが訓練データにうまく適合していることを意味します。

        Iterations: アルゴリズムが最適な解にたどり着くために、パラメータの調整を繰り返した回数です。この場合、1回しか繰り返していません。
        これは、初期パラメータがすでに最適解に非常に近かったか、あるいはアルゴリズムが何らかの理由で早めに停止したことを示唆しています。

        Function evaluations: 最適化プロセス中に、目的関数の値が計算された回数です。

        Gradient evaluations: 最適化プロセス中に、目的関数の勾配が計算された回数です。
        勾配は、アルゴリズムが次にどの方向にパラメータを調整すべきかを決定するために使用されます。

        警告メッセージと最適化結果
        Desired error not necessarily achieved due to precision loss.
        OptimizeWarning: Desired error not necessarily achieved due to precision loss.
        この警告は、最適化が完全には収束しなかったことを示しています。アルゴリズムが求める精度（例えば、勾配のノルムが非常に小さな値になること）に到達する前に、計算上の浮動小数点精度の限界に達してしまい、それ以上改善できなくなったために停止したことを意味します。これは一般的な現象であり、ほとんどの場合、見つかったパラメータは十分に良いものです。

        Optimized parameters: これが最終的な結果です。最適化によって見つかった、最も良い3つのパラメータ (tau, sigma, eta) の値です。これらの値は、プログラム内でexp()関数を使って元の値に戻され、最終的なモデルに適用されます。

    Args:
        xtrain (np.ndarray): 訓練データ点
        ytrain (np.ndarray): 訓練データの値
        kernel_func (function): カーネル関数
        kgrad_func (function): 勾配関数
        init (np.ndarray): 初期パラメータ

    Returns:
        np.ndarray: 最適化されたパラメータ
    """
    res = minimize(loglik, init, args=(xtrain, ytrain, kernel_func, kgrad_func),
                   jac=gradient,
                   method='BFGS',
                   options={'gtol': 1e-4, 'disp': True})
    print(res.message)
    plt.plot(loglik_result_list, marker="o")
    plt.show()
    return res.x

def gpplot(xtrain, ytrain, kernel, params):
    """
    ガウス過程回帰の結果をプロットします。

    Args:
        xtrain (np.ndarray): 訓練データ点
        ytrain (np.ndarray): 訓練データの値
        kernel (function): カーネル関数
        params (np.ndarray): パラメータ
    """
    xx = np.linspace(xmin, xmax, N) # 長さNの等差数列(このテストデータの結果をプロット)
    ypr, spr = gpr(xx, xtrain, ytrain, kernel(params))

    plt.figure(figsize=(10, 6))

    plt.plot(xtrain, ytrain, 'bx', markersize=16, label='Training Data')
    plt.plot(xx, ypr, 'b-', label='Prediction Mean')

    plt.fill_between(xx, ypr - 2 * sqrt(spr), ypr + 2 * sqrt(spr), color=blue, alpha=0.5, label='95% Confidence Interval')

    plt.legend()
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gaussian Process Regression with Optimization')
    simpleaxis()
    plt.show()

def simpleaxis():
    """
    グラフの軸をシンプルにします。
    """
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def main():
    """
    メイン関数
    """
    # ソースコード内でファイル名を直接指定
    data_file = 'data.csv'

    try:
        # np.genfromtxtを使用して、CSVファイルを読み込みます
        train_data = np.genfromtxt(data_file, delimiter=',', dtype=float)
    except IOError:
        print(f"Error: Unable to open file '{data_file}'. Please make sure it exists.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    train_data = train_data[:10, :] # データ数検証用

    xtrain = train_data.T[0]
    ytrain = train_data.T[1]

    # カーネルのパラメータの初期値
    tau_init = log(1)
    sigma_init = log(1)
    eta_init = log(1e-4) # ノイズ項の初期値を小さな値に設定
    params_init = np.array([tau_init, sigma_init, eta_init])

    kernel = kgauss
    kgrad = kgauss_grad

    # 最適化を実行 ガウス過程回帰モデルのハイパーパラメータを最適化するために、対数尤度を最小化！！！
    print('Starting parameter optimization...')
    optimized_params = optimize(xtrain, ytrain, kernel, kgrad, params_init)
    print('Optimized parameters:', exp(optimized_params)) # tau, sigma, eta の順

    gpplot(xtrain, ytrain, kernel, optimized_params)# ガウス過程回帰！！！ 導出したハイパーパラメータを使う！！！

if __name__ == "__main__":
    main()
