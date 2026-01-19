

import numpy as np


def get_cluster_num(datas):
    dists = getDistanceMatrix(datas)
    dc = select_dc(dists)
    rho = get_density(dists, dc)
    deltas, nearest_neiber = get_deltas(dists, rho)
    centers = find_centers_auto(rho, deltas)
    return len(centers)


def get_cluster_DPC(datas):
    dists = getDistanceMatrix(datas)
    percent = 2
    while True:
        dc = select_dc(dists, percent)
        rho = get_density(dists, dc)
        deltas, nearest_neiber = get_deltas(dists, rho)
        centers = find_centers_auto(rho, deltas)
        if len(centers) >= 2:
            labs = cluster_PD(rho, centers, nearest_neiber)
            return labs, len(centers)
        else:
            if percent <= 80:
                percent = percent + 1
            else:
                labs = cluster_PD(rho, centers, nearest_neiber)
                return labs, len(centers)


def getDistanceMatrix(datas):
    # print(np.shape(datas))
    N, D = np.shape(datas)
    dists = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            vi = datas[i, :]
            vj = datas[j, :]
            dists[i, j] = np.sqrt(np.dot((vi - vj), (vi - vj)))
    return dists



def select_dc(dists, percent=2):
    '''算法1'''
    N = np.shape(dists)[0]
    tt = np.reshape(dists, N * N)


    position = int(N * (N - 1) * percent / 100)
    # print("position + N:", position + N)
    dc = np.sort(tt)[position + N]
    return dc


# 计算每个点的局部密度
def get_density(dists, dc, method=None):
    '''
        方法一：截断核
        方法二：高斯核
    '''
    N = np.shape(dists)[0]
    rho = np.zeros(N)
    for i in range(N):
        if method == None:
            rho[i] = np.where(dists[i, :] < dc)[0].shape[0] - 1
        else:
            rho[i] = np.sum(np.exp(-(dists[i, :] / dc) ** 2)) - 1
    return rho



def get_deltas(dists, rho):
    N = np.shape(dists)[0]
    deltas = np.zeros(N)
    nearest_neiber = np.zeros(N)

    index_rho = np.argsort(-rho)
    for i, index in enumerate(index_rho):

        if i == 0:
            continue
        index_higher_rho = index_rho[:i]

        deltas[index] = np.min(dists[index, index_higher_rho])

        index_nn = np.argmin(dists[index, index_higher_rho])
        nearest_neiber[index] = index_higher_rho[index_nn].astype(int)
    deltas[index_rho[0]] = np.max(deltas)
    return deltas, nearest_neiber



def find_centers_auto(rho, deltas):
    rho_threshold = (np.min(rho) + np.max(rho)) * 0.12
    delta_threshold = (np.min(deltas) + np.max(deltas)) * 0.12
    N = np.shape(rho)[0]
    centers = []
    for i in range(N):
        if rho[i] >= rho_threshold and deltas[i] > delta_threshold:
            centers.append(i)
    return centers



def find_centers_K(rho, deltas, K):
    rho_delta = rho * deltas
    centers = np.argsort(-rho_delta)
    return centers[:K]


def cluster_PD(rho, centers, nearest_neiber):
    K = np.shape(centers)[0]
    if K == 0:
        print("can not find centers")
        return
    N = np.shape(rho)[0]
    labs = -1 * np.ones(N).astype(int)
    for i, center in enumerate(centers):
        labs[center] = i
    index_rho = np.argsort(-rho)
    for i, index in enumerate(index_rho):

        if labs[index] == -1:

            labs[index] = labs[int(nearest_neiber[index])]
    return labs
