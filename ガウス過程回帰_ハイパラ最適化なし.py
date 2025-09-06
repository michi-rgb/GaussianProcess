# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 15:07:45 2025

@author: user
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, sqrt
from numpy.linalg import inv

# plot parameters
N = 100
xmin = -1
xmax = 3.5
ymin = -1
ymax = 3

# ガウス過程回帰モデルのハイパーパラメータ
# カーネルのハイパーパラメータ
tau = 1.9    # 1.0 1.9    カーネルの振幅
sigma = 0.6  # 1.0 0.6    カーネルの幅
# カーネル以外のハイパーパラメータ
eta = 0.001  # 0.1 0.0002 ノイズの大きさ


def kgauss(tau, sigma):
    """
    ガウスカーネルを定義します。

    Args:
        tau (float): カーネルの振幅
        sigma (float): カーネルの幅

    Returns:
        function: ガウスカーネル関数
    """
    return lambda x, y: tau * exp(-(x - y)**2 / (2 * sigma * sigma))

def kv(x, xtrain, kernel):
    """
    テストデータと訓練データの間のカーネル値を計算します。

    Args:
        x (float): テストデータ点
        xtrain (np.ndarray): 訓練データ点
        kernel (function): カーネル関数

    Returns:
        np.ndarray: カーネル値のベクトル
    """
    return np.array([kernel(x, xi) for xi in xtrain])

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
    K = np.array([kernel(xi, xj) for xi in xx for xj in xx]).reshape(N_data, N_data)
    return K + eta * np.eye(N_data)

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

    # Kをヒートマップで可視化
    plt.figure(figsize=(8, 6))
    plt.imshow(K, cmap='viridis', origin='lower')
    plt.colorbar(label='Kernel Value')
    plt.title('Kernel Matrix Heatmap')
    plt.xlabel('Training Data Index')
    plt.ylabel('Training Data Index')
    plt.show()

    ypr = []
    spr = []

    for x in xx: # xxは長さNの等差数列(このテストデータの結果をプロット)
        s = kernel(x, x) + eta
        k = kv(x, xtrain, kernel) # テストデータと訓練データの間のカーネル値のベクトルを計算

        ypr.append(k.T.dot(Kinv).dot(ytrain)) # 予測平均
        spr.append(s - k.T.dot(Kinv).dot(k)) # 予測分散

    return np.array(ypr), np.array(spr)

def gpplot(xx, xtrain, ytrain, kernel, params):
    """
    ガウス過程回帰の結果をプロットします。

    Args:
        xx (np.ndarray): 予測を行うテストデータ点
        xtrain (np.ndarray): 訓練データ点
        ytrain (np.ndarray): 訓練データの値
        kernel (function): カーネル関数
        params (list): カーネルパラメータ
    """
    kernel_func = kernel(*params)
    ypr, spr = gpr(xx, xtrain, ytrain, kernel_func) # ガウス過程回帰！！！

    plt.figure(figsize=(10, 6))
    # 訓練データをプロット
    plt.plot(xtrain, ytrain, 'bx', markersize=16, label='Training Data')

    # 予測された平均をプロット
    plt.plot(xx, ypr, 'b-', label='Prediction Mean')

    # 予測の信頼区間（95%）を塗りつぶし
    plt.fill_between(xx, ypr - 2 * sqrt(spr), ypr + 2 * sqrt(spr), color='#ccccff', alpha=0.5, label='95% Confidence Interval')

    plt.legend()
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gaussian Process Regression')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.show()

def usage():
    """
    プログラムの使用方法を表示します。
    """
    print('usage: python3 gpr_refactored.py data.csv [output_image.png]')
    print('$Id: gpr_refactored.py,v 1.1 2024/05/29 11:35:00 gemini Exp $')
    sys.exit(0)

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

    xtrain = train_data.T[0]
    ytrain = train_data.T[1]

    # カーネル関数の定義を簡素化
    kernel = kgauss
    params = [tau, sigma]

    xx = np.linspace(xmin, xmax, N) # 長さNの等差数列(このテストデータの結果をプロット)

    gpplot(xx, xtrain, ytrain, kernel, params) # ガウス過程回帰！！！

if __name__ == "__main__":
    main()
