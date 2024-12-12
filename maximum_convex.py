import sys
from itertools import islice
import numpy as np

ind_t = np.uint8
pos_t = np.int32
tangent_t = np.float32

kSizeArr = 256
kShift = 16
kMask = (1 << kShift) - 1
kPruneStep = 8

points = np.zeros(kSizeArr, dtype=pos_t)
tangents = np.zeros(kSizeArr, dtype=tangent_t)
tmp = np.zeros(kSizeArr, dtype=ind_t)
sorted_arr = np.zeros((kSizeArr, kSizeArr), dtype=ind_t)
lefts = np.zeros((kSizeArr, kSizeArr), dtype=ind_t)
memorize = np.zeros((kSizeArr, kSizeArr), dtype=ind_t)
n = 0

def read_input(m, ps):
    global n, points
    data = ps
    n = m
    index = 1
    for i in range(n):
        x = data[index]
        y = data[index + 1]
        index += 2
        points[i] = (x << kShift) + kMask - y

def order():
    global points
    points[:n] = np.sort(points[:n])
    for i in range(n):
        points[i] = (points[i] & ~kMask) + kMask - (points[i] & kMask)

def preprocess():
    global points, sorted_arr, lefts, tangents, tmp
    for i_center in range(n):
        ind = 0
        x_center = points[i_center] >> kShift
        y_center = points[i_center] & kMask
        nm1 = n - 1
        tmp[:n] = np.arange(n, dtype=ind_t)

        for i_pt in range(n):
            x = points[i_pt] >> kShift
            y = points[i_pt] & kMask
            tangents[i_pt] = tangent_t(y - y_center) / tangent_t(x - x_center)

        tmp[i_center] = tmp[nm1]
        tangents[i_center] = tangents[nm1]

        tmp[:nm1] = sorted(tmp[:nm1], key=lambda l: tangents[l])

        def mk_left(ii):
            lefts[tmp[ii], i_center] = ind

        def mk_sorted(ii):
            nonlocal ind
            sorted_arr[i_center, ind] = tmp[ii]
            ind += 1

        for i in range(nm1):
            if (points[tmp[i]] >> kShift) < x_center:
                mk_left(i)
            else:
                mk_sorted(i)

        for i in range(nm1):
            if (points[tmp[i]] >> kShift) >= x_center:
                mk_left(i)
            else:
                mk_sorted(i)

        if (points[tmp[nm1 - 1]] >> kShift) == x_center and (points[tmp[nm1 - 1]] & kMask) > y_center:
            lefts[tmp[nm1 - 1], i_center] = 0

        sorted_arr[i_center, ind] = ind_t(i_center)

def prune(frm, to):
    global tmp, sorted_arr, lefts
    for i in range(to, n):
        new_j = 0
        delta = 0
        for j in range(n - frm):
            tmp[j] = delta

            if sorted_arr[i, j] < to:
                delta += 1
            else:
                sorted_arr[i, new_j] = sorted_arr[i, j]
                new_j += 1

        for j in range(to, n):
            lefts[j, i] -= tmp[lefts[j, i]]

def dp_step(i_left, frm, to_ind):
    global memorize, sorted_arr, lefts
    child = sorted_arr[frm, to_ind]

    while child < i_left:
        to_ind += 1
        child = sorted_arr[frm, to_ind]

    if child == i_left:
        return 1

    if memorize[frm, child]:
        return memorize[frm, child]

    x = max(dp_step(i_left, child, lefts[frm, child]) + 1, dp_step(i_left, frm, to_ind + 1))

    memorize[frm, child] = x
    return x

def dp(i_left):
    return dp_step(i_left, i_left, 0)

def solve():
    global memorize
    best = 0
    i_prune = kPruneStep
    for i_left in range(n - best):
        memorize.fill(0)
        best = max(best, dp(i_left))

        if i_prune == 1:
            i_prune = kPruneStep
            prune(i_left + 1 - kPruneStep, i_left + 1)
        else:
            i_prune -= 1

    return best

def main(m,ps):
    read_input(m,ps)
    order()
    preprocess()
    return solve()

if __name__ == "__main__":
    main()
