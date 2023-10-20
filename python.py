import collections

strs = ["eat","tea","tan","ate","nat","bat"]

ans = collections.defaultdict(list)

for s in strs:
    count = [0] * 26
    for c in s:
        index = ord(c) - ord("a")
        count[index] += 1
    ans[tuple(count)].append(s)
    print(ans,"\n")
ans.values()
