from typing import List, Tuple

class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        m , n = len(matrix), len(matrix[0])
        zero_rows = set()
        zero_cols = set()

        ##first pass
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    zero_rows.add(m)
                    zero_cols.add(n)
        
        #second pass: set selected rows to 


class Matrix:
    def deg_90_rotate(self, matrix: List[List[int]]) -> None:
        n = len(matrix)

        ##rotate by row
        for i in range(n // 2):
            j = n - i - 1
            matrix[i], matrix[j] = matrix[j], matrix[i]

        ##swap elements across diagonal
        for i in range(n):
            for j in range(i+1, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]


##implementation of a time based kv cache
class TimeMap:
    def __init__(self):
        self.time_series = defaultdict(list)

    def set(self, key: str, val: str, timestamp: int):
        self.time_series[key].append([val, timestamp])

    def get(self, key, timestamp):
        stored_values : List[Tuple[int, str]] = self.time_series[key]
        latest_valid_value: str = " "

        left, right = 0, len(stored_values) - 1
        while left <= right:
            current_value, current_timestamp = stored_values[key]
            mid = (left + right) // 2
            if current_timestamp <= timestamp:
                latest_valid_value = current_value
                left = mid + 1
            else:
                right = mid - 1

        return latest_valid_value
             

##obv you should be able to modify this to extract values and other things
def search_matrix(matrix: List[List[int]], target: int) -> bool:
    if not matrix or not matrix[0]:
        return False
    m, n = len(matrix), len(matrix[0])
    left, right = 0, m * n - 1

    while left <= right:
        mid = (left + right) // 2
        row, col = mid // n, mid % n
        value = matrix[row][col]

        if value == target:
            return True
        elif value < target:
            left = mid + 1
        else:
            right = mid - 1
    return False

def search_range(arr: List[int], target: int) -> List[int]:
    ##we just have to find the bounds of the target
    def find_bound(n: bool):
        left, right = 0, n - 1
        result = -1

        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                result = mid
                if n:
                    right = mid - 1
                else:
                    left = mid + 1
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return result
    return [find_bound(True), find_bound(False)]

def search_rotated(arr: list[int], target: int) -> int:
    left, right = 0, len(arr) - 1

    if len(arr) == 0:
        return -1

    ##figure out which half is rotated
    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid

        if arr[left] <= arr[mid]:
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
            
        else:
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid + 1

    return -1


# Finds index of element closest to target in a sorted array
def find_closest(arr: list[int], target: int) -> int:
    if len(arr) == 1:
        return 0
    
    left, right = 0, len(arr) - 1

    while left + 1 <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid
        elif arr[mid] > target:
            right = mid

    #compare abs values of the last two values; the arr only contains two values atp
    if abs(arr[left] - target) <= abs(arr[right] - target):
        return left
    else:
        return right

## find the kth largest element in sorted array
from typing import List
import random

def findKthLargest(nums: List[int], k: int) -> int:
    def quickSelect(nums, k):
        pivot = random.choice(nums)

        ##partion into three
        left = [x for x in nums if x > pivot]
        mid = [x for x in nums if x == pivot]
        right = [x for x in nums if x < pivot]

        ##you search through the partitions recursively for k
        ##it resolves to the pivot point
        if len(left) < k:
            return quickSelect(left, k)
        elif len(left) + len(mid) < k:
            return quickSelect(right, k - len(left) - len(mid))
        else:
            return pivot
    
    return quickSelect(nums, k)




def binary_search(nums: List[int], k:int) -> int:
    start, end = 0, len(nums) - 1

    while start <= end:
        mid = (start + end) // 2
        if nums[mid] < k:
            start = mid + 1
        elif nums[mid] > k:
            end = mid - 1
        elif nums[mid] == k:
            return mid
        
    return -1 

        
print(binary_search([1,2,3,4,5], 3))



## trapping rain water in boundaries
from typing import List

def trap(height: List[int]) -> int:

    n = len(height)
    if n <= 2:
        return 0
        
    l, r = 0, n - 1  # Left and right pointers
    max_l = height[0]  # Maximum height from left
    max_r = height[n-1]  # Maximum height from right
    result = 0
 
    while l < r:
        if height[l] < height[r]:
            # If left height is smaller, process left side
            max_l = max(max_l, height[l])
            result += max_l - height[l]  # Add trapped water
            l += 1
        else:
            # If right height is smaller or equal, process right side
            max_r = max(max_r, height[r])
            result += max_r - height[r]  # Add trapped water
            r -= 1

    return result

# Test the function
test_height = [0,1,0,2,1,0,1,3,2,1,2,1]
print(f"Water trapped: {trap(test_height)}")  # Output: 6







## check the inclusion of string 1 inside of string 2
def checkIncl(s1: str, s2: str):
    ## count s1 char frequencies
    ## create sliding window to track freq in s2
    ## resize if window exceeds s1 

    s1_count = [0] * 26







## topk frequent (hashmap)
from collections import defaultdict
from typing import List
def topKFrequent(nums: List[int], k: int) -> List[int]:
    ## count elements using a hash map
    freq = defaultdict(int)
    n = len(nums)
    for num in nums:
        freq[num] += 1
    
    buckets = [[] for _ in range(n+1)]

    for num, frequency in freq.items(): ## dict contains nums as keys an+d freq as val
        buckets[frequency].append(num) ##basically swap keys and values into buckets
        print(buckets[frequency])

    result = []
    while n > -1 and k > 0:
        if buckets[n]:
            result.append(buckets[n].pop())
            k -= 1

            
def merge(nums1, m, nums2, n):
    p1 = m -1 
    p2 = n - 1
    p =  m + n - 1

    while p2 >= 0:
        if p1 >= 0 and nums1[p1] > nums2[p2]:
            ## place the appropriate sized p1 el at the pointer
            nums1[p] = nums1[p1]
            p1 += 1
        else:
            ##else place the appropriately sized p2 el at the pointer
            nums1[p] = nums2[p2]
            p2 -= 1
        p -= 1



# from typing import List, Set, Tuple

# def three_sum(nums: List[int]) -> List[Tuple[int, int, int]]:

#     ## create triplets and sort in place
#     triplets = set()
#     nums.sort()

#     for i, n in enumerate(nums):
#         if n > 0:
#             break

#         if i > 0 and nums[i - 1] == nums[i]:
#             continue

#         ## start two pointer process, and account for duplicates
#         ## this is due to skipping duplicates on the first number
#         l, r = i + 1, len(nums) - 1
#         while l < r:
#             current_sum = n + nums[l] + nums[r]
#             if current_sum == 0:
#                 triplets.add(n, nums[l], nums[r])
#                 ##find the next triplet
#                 left += 1
#                 right -= 1
#             if nums[l] < nums[r] and nums[l] == nums[l - 1]:
#                 l += 1
#             if nums[l] < nums[r] and nums[r] == nums[r + 1]:
#                 r -= 1
#             elif current_sum < 0:
#                 l += 1
#             else:
#                 r -= 1
#         return list(triplets)
# def max_area(height: List[int]) -> int:
#     n = len(height)
#     l, r = 0, n - 1
#     res = 0

#     while l < r:
#         max_area = min(height[l] , height[r]) * (r - l)
#         ##update if curr is larger
#         res = max(res, max_area)

#         if height[l] < height[r]:
#             l += 1
#         else: 
#             r -= 1
#     return res

# from typing import List

# def trap(height: List[int]) -> int:
#     n = len(height)
#     l , r = 0, n- 1
#     max_l = height[l]
#     max_r = height[r]

#     res = 0
#     ## if height only contain two, no water is trapped
#     if n <= 2:
#         return 0
#     while l < r:
#         if height[l] < height[r]:
#             max_l = max(max_l, height[l])
#             ## calculate trapped water
#             res += max_l - height[l]
#             l += 1
#         else:
#             ##case in which right is greater than or equal 
#             max_r = max(max_r, height[r])
#             res += max_r - height[r]
#             r += 1

#         return res










# ## check the inclusion of string 1 inside of string 2
# def checkIncl(s1: str, s2: str):
#     s1_count = [0] * 26
#     left = 0

#     for char in s1:
#         s1_count[ord(char) - ord('a')] += 1

#     s2_count = [0] * 26
#     ## increasing the window
#     for right in range(len(s2)):
#         s2_count[ord(s2[right]) - ord['a']] += 1
        
#         ## resizing when window exceeds s1, we add one cause our window bounds are 0 indexed
#         if right - left + 1 > len(s1):
#             s1_count[ord(s1[right]) - ord('a')] -= 1
#             left += 1

#         if s1_count == s2_count:
#             return True

#     return False
    
    

# # from collections import deque

# # def maxslidingg(nums, k):
# #     q = deque() ## this deque is used to store indices-
# #     l = r = 0

# #     def slide_right():
# #         nonlocal r
# #         if q and nums[q[-1]] < nums[r]:
# #             q.pop()
# #         q.append(r)
# #         r += 1

    
# #     def slide_left():
# #         nonlocal l
# #         l += 1
# #         if q and l > q[0]:
# #             q.popleft()
    
# #     result = []
# #     while r < k:
# #         slide_right()
# #     result.append(nums[q[0]])

# #     while r < len(nums):
# #         slide_right()
# #         slide_left()
# #         result.append(nums[q[0]])
    
# #     return result

















# # ## find the max subarray of size k

# # def findMaxAvg(nums, k):
# #     ## init window of size k
# #     ## track max sum until array ends
# #     ## return sum / k (avg)

# #     current_sum = sum(nums[:k])
# #     max_sum = current_sum

# #     ## start from k, stop at len(nums)
# #     for i in range(k, len(nums)):
# #         current_sum = current_sum - nums[i - k] + nums[i]
# #         max_sum = max(max_sum, current_sum)

# #     return max_sum / k



# # from collections import defaultdict

# # def characterReplacement(s: str, k: int) -> int:
# #     ##read strings into asso array
# #     ##slide window across keeping track of sz using boundaries
# #     ##keep track of the max sub
# #     sub = defaultdict(int)
# #     max_count = 0
# #     l = r = 0

# #     ##how to properly slide
# #     while r < len(s):
# #         sub[s[r]] += 1
# #         max_count = max(max_count, sub[s[r]])

# #         ## if characters to replace exceeds k, resz window
# #         if right - left - max_count > k:
# #             l += 1
# #             sub[s[l]] -= 1

# #     return right - left






















# # ##count palindromic substrings
we use manacher
# # def count_substrings(s : str) -> int:
# #     n = len(s)
# #     new_str = "#" + "#".join(s) + "#"
# #     p = [0] * n ## palindromic radii
# #     center = right = 0
    
# #     while i < range(n):
# #         ## take advantage of mirror property if i < right
# #         if i < right:
# #             mirror = 2 * center - i
# #             p[i] = min(right - i, p[mirror])


# #         ##expanding around i
# #         left = i - (p[i] + 1)
# #         r = i + (p[i] + 1)
# #         while left >= 0 and right < n and new_str[left] == new_str[right]: ## bcos palindromic
# #             p[i] = 1 ##expand palindromic radii and bounds
# #             left -= 1
# #             r += 1
        

# #         ##updating center and right bound if necessary
# #         if i + p[i] > right:
# #             center = i
# #             right = i + p[i]

# #         return sum((v + 1) // 2 for v in p if v > 0)


# # ##maxslidingwindow
# # def maxsliding(nums: List[int], k: int) -> List[int]:
# #     q = deque() ## remember: q is used to store indices
# #     left = right = 0

# #     def slide_right():
# #         nonlocal right
# #         while q and nums[q[-1]] < nums[right]:
# #             q.pop()
# #         q.append(r)
# #         right += 1
    
# #     def slide_left():
# #         nonlocal left
# #         l += 1
# #         while q and l > q[0]:
# #             q.popleft()
        

# #     results = []
# #     ##load up window to first size k
# #     while r < k:
# #         slide_right()
# #     results.append(nums[q[0]])

# #     ##take care of the rest of the windows
# #     while r < len(nums):
# #         slide_right()
# #         slide_left()
# #         results.append(nums[q[0]])
    
# #     return result



# # ##character replacement
# # #longest repeating character replacement
# # from collections import defaultdict
# # def characterReplacement(s: str, k :int):
# #     count = defaultdict(s)
# #     left = right = 0

# #     max_count = 0

add new character to window
check if character has highest freq, update the max count if so
resize window if character outside window
    while right < len(s):
        count[s[right]] += 1
        max_count = max(max_count, count[s[right]])
        right += 1

    if right - left - max_count > k:
        ## mov left pos by 1
        left += 1
        count[s[left]] -= 1

    return right - left

#diff btwn dict and defaultdict is that defaultdict generates keys for ones that do not have

##length of longest substring without repeating
def longestSubstring(s: str):
    #define a dict to store the index of each character
    # if the character is seen and at the start of the index window, start over (place start at curr idx)
    char_index = {}
    start = 0
    max_length = 0

    for end, char in enumerate(s):
        if char is in char_index and char_index[char] >= start:
            start = char_index[char] + 1
        else:
            ## we are basically calculating the longest substring in the window, so when we see the first unique character = start over
            max_length = max(max_length, end - start + 1 )

        char_index[char] = end

    return max_length



##max subarray avg (same with max subarray sum let's be honest)

def maxAvg(nums: list[float], k: int) -> int :
    curr_sum = sum(nums[:k]) ##partial sum up to k that is
    max_sum = curr_sum
    
    for i in range(k, len(nums)):
        curr_sum = curr_sum - nums[i - k] + nums[i] ## we add nums i to add the number at the window
        max_sum = max(max_sum, curr_sum)

    return max_sum / k





## topk frequent (hashmap)
from collections import defaultdict
from typing import List
def topKFrequent(nums: List[int], k: int) -> List[int]:
    ## count elements using a hash map
    freq = defaultdict(int)
    n = len(nums)
    for num in nums:
        freq[num] += 1
    
    buckets = [[] for _ in range(n+1)]

    for num, frequency in freq.items(): ## dict contains nums as keys an+d freq as val
        buckets[frequency].append(num) ##basically swap keys and values into buckets
        print(buckets[frequency])

    result = []
    while n > -1 and k > 0:
        if buckets[n]:
            result.append(buckets[n].pop())
            k -= 1
        else:
            n -= 1

    return result


print(topKFrequent([1,1,1,2,2,2,2,2,3,3,4,4,4], 2))


## happy number (hashmap)
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










# 1. break down arr into small bits by midpoint
# 2. sort when we get to base case
# 3. rebuild sorted arr
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left_arr = arr[:mid]
    right_arr = arr[mid:]
    return merge(left_arr, right_arr)

def merge(left_arr, right_arr) -> List[arr]:
        i = 0, j = 0
        result = []
        while i < len(left_arr) and j < len(right_arr):
            if left_arr[i] < right_arr[j]:
                result.append(left_arr[i])
                i += 1
            else:
                result.append(right_arr[j])
                j += 1

    ##append very last el
    result.extend(left_arr[i:])
    result.extend(right_arr[j:])

    return result   


##first non-repeating character in a string (sorting)
## the counter data struct basically uses values as keys and stores frequencies as values.
from defaultdict import Counter
def firstUnique(s):
    count = Counter(s)

    for c in count.keys():
        if count[c] == 1:
            return count.index(c)
    
    return -1


def two_sum(nums: List[int], target: int) -> List[int]:
## find the indices of two numbers that add up to the target
    seen = {}

    ## enumerate is the fn that returns index + val
    for i, n in enumerate(nums):
        complement = target - n
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i

    return []

##intersection of sets
def intersection(num1: List[int], num2: List[int]) -> List[int]:
    ## dont do in place.
    set1 = set(num1)

    return [x for x in set1 if x in set(num2)]

## contains duplicate characters
def containsDuplicate()

def containsDuplicate()

























# ##merge-sort
# from typing import List
# def merge_sort(arr: List[int]) -> List[int]:
#     if len(arr) <= 1:
#         return arr
    
#     ##find midpoint of arr, split down the middle
#     mid = len(arr) // 2
#     left = merge_sort(arr[:mid])
#     right = merge_sort(arr[mid:])

#     return merge(left,right)
    
# def merge(left: List[int], right: List[int]) -> List[int]:
#     i = j = 0
#     result = []

#     while i < len(left) and j < len(right):
#         #split in two, then compare and sort the elements of the split array one by one
#         if left[i] < right[j]:
#             result.append(left[i])
#             i += 1
#         else:
#             result.append(right[j])
#             j += 1

#     result.extend(left[i:])
#     result.extend(right[j:])    

#     return result



# new_arr = merge_sort([2,3,5,6,1,3,9])
# print(new_arr)

# from typing import Counter

# def first_non_repeating(s: str) -> int:
#     count = Counter(s)

#     for c in count.keys():
#         if count[c] == 1:
#             return s.index(c)
#     return -1


# def two_sum(nums: list[int], target: int) -> list[int]:

#     seen = { }
#     for i, num in enumerate(nums):
#         complement = target - num
#         if complement in seen:
#             return [seen[complement], i]
#         seen[num] = i

#     return []

def intersection(num1: int, num2: int) -> list[int]:
    set1 = set(num1)

    return [x for x in set1 if x in set(num2)]

class Solution:
    def isHappy(self, n: int) -> bool:
        visited = set()

        while n not in visited:
            visited.add(n)

            if n == 1:
                return True
            
            current_sum : int = 0
            while n > 0:
                digit =  n % 10
                current_sum += digit * digit
                n //= 10
            
            n = current_sum
        
        return False
    
result = Solution.isHappy(Solution, 19)
print(result)

from typing import List
from collections import defaultdict

def groupAnagrams(strs: List[int]) -> List[List[int]]:
    res = defaultdict(list)

    for s in strs:
        ##create an array to count char frequency
        count = [0] * 26

        ##count freq in string, set alphabet as index
         for c in s:
            count[ord(c) - ord["a"]] += 1

        res[tuple[count]].append(s)
    return list(res.values())
