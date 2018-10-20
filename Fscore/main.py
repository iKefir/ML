N = int(input())

conf_matrix = [[] for i in range(N)]
tps = [0 for i in range(N)]
fps = [0 for i in range(N)]
fns = [0 for i in range(N)]

for i in range(N):
    conf_matrix[i] = list(map(int, input().split()))

for i in range(N):
    for j in range(N):
        if i == j:
            tps[i] += conf_matrix[i][j]
        else:
            fps[i] += conf_matrix[i][j]
            fns[j] += conf_matrix[i][j]

precision = [tp / (tp + fp) if (tp + fp != 0) else 0 for tp, fp in zip(tps, fps)]
recall = [tp / (tp + fn) if (tp + fn != 0) else 0 for tp, fn in zip(tps, fns)]

f_score = [2 * pr * rc / (pr + rc) if (pr + rc != 0) else 0 for pr, rc in zip(precision, recall)]

els = [tp + fp for tp, fp in zip(tps, fps)]

wgh_macro_f_score = sum([f_s * el for f_s, el in zip(f_score, els)]) / sum(els) if sum(els) != 0 else 0

wgh_precision = sum([pr * el for pr, el in zip(precision, els)]) / sum(els) if sum(els) != 0 else 0
wgh_recall = sum([rc * el for rc, el in zip(recall, els)]) / sum(els) if sum(els) != 0 else 0

wgh_micro_f_score = 2 * wgh_precision * wgh_recall / (wgh_precision + wgh_recall) if (wgh_precision + wgh_recall != 0) else 0

print(wgh_micro_f_score)
print(wgh_macro_f_score)
