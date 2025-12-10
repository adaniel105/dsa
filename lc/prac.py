# from collections import defaultdict
# from typing import List
# def topKFrequent(nums: List[int], k: int) -> List[int]:
#     ## count elements using a hash map
#     freq = defaultdict(int)
#     n = len(nums)
#     for num in nums:
#         freq[num] += 1
    
#     buckets = [[] for _ in range(n+1)]

#     for num, frequency in freq.items(): ## dict contains nums as keys an+d freq as val
#         buckets[frequency].append(num) ##basically swap keys and values into buckets
#         print(buckets[frequency])

#     result = []
#     while n > -1 and k > 0:
#         if buckets[n]:
#             result.append(buckets[n].pop())
#             k -= 1
#         else:
#             n -= 1

#     return result


# print(topKFrequent([1,1,1,2,2,2,2,2,3,3,4,4,4], 2))


def isHappy(n):

    visited = set()

    while n not in visited:
        visited.add(n)

        if n == 1:
            return True
        current_sum  = 0
        ## simulate the happy number sequence
        while n > 0:
            digit = n % 10
            current_sum += digit * digit
            n //= 10
        
        ## dont forget to replace
        n = current_sum
    return False

print(isHappy(19))










# # 1. break down arr into small bits by midpoint
# # 2. sort when we get to base case
# # 3. rebuild sorted arr
# def merge_sort(arr):
#     if len(arr) <= 1:
#         return arr
    
#     mid = len(arr) // 2
#     left_arr = arr[:mid]
#     right_arr = arr[mid:]
#     return merge(left_arr, right_arr)

# def merge(left_arr, right_arr) -> List[arr]:
#         i = 0, j = 0
#         result = []
#         while i < len(left_arr) and j < len(right_arr):
#             if left_arr[i] < right_arr[j]:
#                 result.append(left_arr[i])
#                 i += 1
#             else:
#                 result.append(right_arr[j])
#                 j += 1

#     ##append very last el
#     result.extend(left_arr[i:])
#     result.extend(right_arr[j:])

#     return result   


# ##first non-repeating character in a string
# ## the counter data struct basically uses values as keys and stores frequencies as values.
# from defaultdict import Counter
# def firstUnique(s):
#     count = Counter(s)

#     for c in count.keys():
#         if count[c] == 1:
#             return count.index(c)
    
#     return -1


# def two_sum(nums: List[int], target: int) -> List[int]:
# ## find the indices of two numbers that add up to the target
#     seen = {}

#     ## enumerate is the fn that returns index + val
#     for i, n in enumerate(nums):
#         complement = target - n
#         if complement in seen:
#             return [seen[complement], i]
#         seen[num] = i

#     return []


# def intersection(num1: List[int], num2: List[int]) -> List[int]:
#     ## dont do in place.
#     set1 = set(num1)

#     return [x for x in set1 if x in set(num2)]


# def containsDuplicate()

























# # ##merge-sort
# # from typing import List
# # def merge_sort(arr: List[int]) -> List[int]:
# #     if len(arr) <= 1:
# #         return arr
    
# #     ##find midpoint of arr, split down the middle
# #     mid = len(arr) // 2
# #     left = merge_sort(arr[:mid])
# #     right = merge_sort(arr[mid:])

# #     return merge(left,right)
    
# # def merge(left: List[int], right: List[int]) -> List[int]:
# #     i = j = 0
# #     result = []

# #     while i < len(left) and j < len(right):
# #         #split in two, then compare and sort the elements of the split array one by one
# #         if left[i] < right[j]:
# #             result.append(left[i])
# #             i += 1
# #         else:
# #             result.append(right[j])
# #             j += 1

# #     result.extend(left[i:])
# #     result.extend(right[j:])    

# #     return result



# # new_arr = merge_sort([2,3,5,6,1,3,9])
# # print(new_arr)

# # from typing import Counter

# # def first_non_repeating(s: str) -> int:
# #     count = Counter(s)

# #     for c in count.keys():
# #         if count[c] == 1:
# #             return s.index(c)
# #     return -1


# # def two_sum(nums: list[int], target: int) -> list[int]:

# #     seen = { }
# #     for i, num in enumerate(nums):
# #         complement = target - num
# #         if complement in seen:
# #             return [seen[complement], i]
# #         seen[num] = i

# #     return []

# def intersection(num1: int, num2: int) -> list[int]:
#     set1 = set(num1)

#     return [x for x in set1 if x in set(num2)]

# class Solution:
#     def isHappy(self, n: int) -> bool:
#         visited = set()

#         while n not in visited:
#             visited.add(n)

#             if n == 1:
#                 return True
            
#             current_sum : int = 0
#             while n > 0:
#                 digit =  n % 10
#                 current_sum += digit * digit
#                 n //= 10
            
#             n = current_sum
        
#         return False
    
# result = Solution.isHappy(Solution, 19)
# print(result)

# from typing import List
# from collections import defaultdict

# def groupAnagrams(strs: List[int]) -> List[List[int]]:
#     res = defaultdict(list)

#     for s in strs:
#         ##create an array to count char frequency
#         count = [0] * 26

#         ##count freq in string, set alphabet as index
#         for c in s:
#             count[ord(c) - ord["a"]] += 1

#         res[tuple[count]].append(s)
#     return list(res.values())
