from operator import itemgetter


N, M, K = map(int, input().split())

classes = zip(range(1, N + 1), list(map(int, input().split())))

classes = sorted(classes, key=itemgetter(1))

ans = [[] for i in range(K)]

ctr = 0
for it in classes:
    ans[ctr].append(it[0])
    ctr = (ctr + 1) % K

for l in ans:
    print(len(l), " ".join(map(str, l)))
