"""

"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division



def minimum_distance(nums, start, target):
    """
    Q1 - Minimum Distance to the Target Element *

    :param nums:
    :param start:
    :param target:
    :return:
    """

    min_abs = abs(0 - start)
    for i, val in enumerate(nums):
        absolute_minus = abs(i - start)
        if val == target:
            if absolute_minus < min_abs: min_abs = absolute_minus

    return min_abs


def SSID(string):
    """
    Q2 - Splitting a String Into Descending Consecutive Values *
    :param string:
    :return:
    """

    def back_tracking(s, i, num, cnt):
        """

        :param s:
        :param i:
        :param num:
        :param cnt:
        :return:
        """

        if len(s) == i:
            return cnt >= 2

        new_num = 0
        for j in range(i, len(s)):
            new_num = new_num * 10 + int(s[j])
            if new_num > num > 0:
                break
            if (num == -1 or num - 1 == new_num) and back_tracking(s, j + 1, new_num, cnt + 1):
                return True
        return False

    return back_tracking(string, 0, -1, 0)


def getMinSwaps(num, k):
    """Q3: Minimum Adjacent Swaps to Reach the Kth Smallest Number *

    :param num:
    :param k:
    :return:
    """

    def next_permutation(nums, begin, end):
        def reverse(nums, begin, end):
            left, right = begin, end - 1
            while left < right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1

        k, l = begin - 1, begin
        for i in reversed(range(begin, end - 1)):
            if nums[i] < nums[i + 1]:
                k = i
                break
        else:
            reverse(nums, begin, end)
            return False
        for i in reversed(range(k + 1, end)):
            if nums[i] > nums[k]:
                l = i
                break
        nums[k], nums[l] = nums[l], nums[k]
        reverse(nums, k + 1, end)
        return True

    new_num = list(num)
    while k:
        next_permutation(new_num, 0, len(new_num))
        k -= 1
    result = 0
    for i in range(len(new_num)):
        if new_num[i] == num[i]:
            continue
        #   // greedily move the one with the least cost from new_num to num without missing optimal cost
        for j in range(i + 1, len(new_num)):
            if new_num[j] == num[i]:
                break
        result += j - i
        for j in reversed(range(i + 1, j + 1)):
            new_num[j], new_num[j - 1] = new_num[j - 1], new_num[j]
    return result


import heapq


def min_interval_included(intervals, queries):
    """

    :param intervals:
    :param queries:
    :return:
    """
    intervals.sort()
    queries = [(q, i) for i, q in enumerate(queries)]
    queries.sort()
    min_heap = []
    i = 0
    result = [-1] * len(queries)
    for q, idx in queries:
        while i != len(intervals) and intervals[i][0] <= q:
            heapq.heappush(min_heap, [intervals[i][1] - intervals[i][0] + 1, i])
            i += 1
        while min_heap and intervals[min_heap[0][1]][1] < q:
            heapq.heappop(min_heap)
        result[idx] = min_heap[0][0] if min_heap else -1
    return result


if __name__ == '__main__':
    # nums = [1, 2, 3, 4, 5]
    # target = 5
    # start = 3
    #
    # print(minimum_distance(nums, start, target))
    # s = "1000908"
    # print(SSID(s))
    #     num = "00123"
    #     print(getMinSwaps(num, k=1))

    intervals = [[2, 3], [2, 5], [1, 8], [20, 25]]
    queries = [2, 19, 5, 22]
    print(min_interval_included(intervals, queries))
