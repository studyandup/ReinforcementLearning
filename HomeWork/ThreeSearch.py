#!/user/bin/env python3
# -*- conding:utf-8 -*-
# @auther Zhang
# @date 2021/9/30
# @file ThreeSearch.py
# 三分查找
def ThreeSearch(List, k, left, right):
    if left >= right:
        return False
    # 确定将区间三等分的2个点
    mid1 = (right - left) // 3 + left
    mid2 = (right - left) * 2 // 3 + left
    # 分5种情况讨论,包括3个区间，2个点
    if List[mid1] == k:
        return mid1
    elif List[mid2] == k:
        return mid2
    elif List[mid1] > k:
        return ThreeSearch(List, k, left=left, right=mid1)
    elif k > List[mid1] and k < List[mid2]:
        return ThreeSearch(List, k, left=mid1 + 1, right=mid2)
    else:
        return ThreeSearch(List, k, left=mid2 + 1, right=right)

if __name__ == '__main__':
    List = [i for i in range(1, 100, 2)]
    print(List)
    print(ThreeSearch(List, 1, 0, len(List)))