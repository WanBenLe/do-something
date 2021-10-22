'''
# 删除排序数组中的重复项
# len(set(nums))
def removeDuplicates(nums):
    for i in range(len(nums) - 1, 0, -1):
        if nums[-i] == nums[-i - 1]:
            del nums[-i]
    return len(nums)


nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
print(removeDuplicates(nums))


# 买卖股票的最大时机 II
def maxProfit(prices):
    a = 0
    for i in range(len(prices) - 1):
        x = prices[i + 1] - prices[i]
        if x > 0:
            a += x
    return a


prices = [7, 6, 4, 3, 1]
print(maxProfit(prices))


# 旋转数组
def rotate(nums, k):
    k %= len(nums)
    nums[:] = nums[-k:] + nums[:-k]
    print(nums)


nums = [1, 2, 3, 4, 5, 6, 7]
k = 3
rotate(nums, k)


# 存在重复元素
def containsDuplicate(nums):
    if len(nums) != len(set(nums)):
        return True
    return False


nums = [1, 1, 1, 3, 3, 4, 3, 2, 4, 2]
print(containsDuplicate(nums))


# 只出现一次的数字
def singleNumber(nums):
    a = nums[0]
    # ^异或位运算,满足交换律,相等为0,否则保留原数
    for i in range(len(nums) - 1):
        a = a ^ nums[i + 1]
    return a


nums = [4, 1, 2, 1, 2]
print(singleNumber(nums))


# 两个数组的交集 II
def intersect(nums1, nums2):
    dict1 = {}
    for i in range(len(nums1)):
        if nums1[i] not in dict1:
            dict1[nums1[i]] = 1
        else:
            dict1[nums1[i]] += 1
    result = []
    for i in range(len(nums2)):
        if nums2[i] not in dict1:
            continue
        elif dict1[nums2[i]] > 0:
            result.append(nums2[i])
            dict1[nums2[i]] -= 1
    return result


nums1 = [4, 9, 5]
nums2 = [9, 4, 9, 8, 4]
print(intersect(nums1, nums2))


# 加一
def plusOne(digits):
    for site in range(len(digits) - 1, -1, -1):
        if digits[site] != 9:
            digits[site] += 1
            return digits
        else:
            digits[site] = 0
    # 全是9补一
    digits.insert(0, 1)
    return digits


digits = [9, 3, 2, 1]
print(plusOne(digits))


# 移动零
def moveZeroes(nums):
    indb = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[i], nums[indb] = nums[indb], nums[i]
            indb += 1
    print(nums)


nums = [0, 1, 0, 3, 12]
moveZeroes(nums)


# 两数和
def twoSum(nums, target):
    d = {}
    for a, b in enumerate(nums):
        if target - b in d.keys():
            return [d[target - b], a]
        else:
            d[b] = a


nums = [2, 7, 11, 15]
target = 9
print(twoSum(nums, target))


# 有效的数独
def isValidSudoku(board):
    a = [0] * 9
    b = [0] * 9
    c = [0] * 9
    for i in range(9):
        for j in range(9):
            if board[i][j] == '.':
                continue
            # 得到的应该是2^int的结果
            x = 1 << int(board[i][j])
            # 重复得到自己,不重复为0
            t1 = (a[i] & x) > 0
            t2 = (b[j] & x) > 0
            # 9宫格取整
            t3 = (c[j // 3 * 3 + i // 3] & x) > 0
            if t1 | t2 | t3:
                return False
            else:
                # 用|记录位置,其实是一个int里面的位数每一个二进制被塞了个1,例如9就是[100000000],1就是[000000001],9和1就是[100000001]
                a[i] = a[i] | x
                b[j] = b[j] | x
                c[j // 3 * 3 + i // 3] = c[j // 3 * 3 + i // 3] | x
    return True


board = [["5", "3", ".", ".", "7", ".", ".", ".", "."]
    , ["6", ".", ".", "1", "9", "5", ".", ".", "."]
    , [".", "9", "8", ".", ".", ".", ".", "6", "."]
    , ["8", ".", ".", ".", "6", ".", ".", ".", "3"]
    , ["4", ".", ".", "8", ".", "3", ".", ".", "1"]
    , ["7", ".", ".", ".", "2", ".", ".", ".", "6"]
    , [".", "6", ".", ".", ".", ".", "2", "8", "."]
    , [".", ".", ".", "4", "1", "9", ".", ".", "5"]
    , [".", ".", ".", ".", "8", ".", ".", "7", "9"]]
print(isValidSudoku(board))


# 旋转图像
def rotate(matrix) -> None:
    # 转列
    for i, vec in enumerate(zip(*matrix)):
        matrix[i] = list(vec)
        # 倒序
        matrix[i].reverse()
    print(matrix)


matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
rotate(matrix)


# 反转字符串
def reverseString(s):
    x = len(s) // 2
    for i in range(x):
        s[i], s[-(i + 1)] = s[-(i + 1)], s[i]
    print(s)


s = ["h", "e", "l", "l", "o"]
reverseString(s)


# 整数反转
def reverse(x):
    fu = 0
    if x < 0:
        fu = 1
        x = -x
    t = 0
    y = 0
    while x > 0:
        y = y * 10 + x % 10
        x = x // 10
        t += 1
    if y > 2 ** 31:
        return 0
    if fu == 1:
        return -y
    else:
        return y


x = -123
print(reverse(x))


# 字符串中的第一个唯一字符
def firstUniqChar(s):
    d = {}
    for i in range(len(s)):
        if s[i] not in d:
            d[s[i]] = 1
        else:
            d[s[i]] += 1
    for i in range(len(s)):
        if d[s[i]] == 1:
            return i
    return -1


s = "loveleetcode"
print(firstUniqChar(s))


# 有效的字母异位词
def isAnagram(s, t):
    d = {}
    e = {}
    for i in s:
        if i not in d:
            d[i] = 1
        else:
            d[i] += 1
    for i in t:
        if i not in e:
            e[i] = 1
        else:
            e[i] += 1
    return d == e


s = "ab"
t = "ba"

print(isAnagram(s, t))


# 验证回文串
def isPalindrome(s):
    s = s.lower()
    # 是字母和是数字的
    s = ''.join([x for x in s if x.isalpha() or x.isdigit()])
    x = len(s) // 2
    for i in range(x):
        if s[i] != s[-(i + 1)]:
            return False
    return True


s = "A man, a plan, a canal: Panama"
print(isPalindrome(s))


# 字符串转换整数 (atoi)
def myAtoi(s):
    indx = 0
    result = ''
    k = 0
    while indx < len(s):
        if s[indx] == ' ' and k == 0:
            indx += 1
            continue
        elif (s[indx] == '+' or s[indx] == '-') and k == 0:
            result += s[indx]
            k = 1
        elif s[indx].isdigit():
            result += s[indx]
            k = 1
        else:
            break
        indx += 1
    if result == '' or result == '+' or result == '-':
        return 0
    result = int(result)
    if result > (2 ** 31 - 1):
        return 2 ** 31 - 1
    elif result < (-2 ** 31):
        return -2 ** 31
    return result


print(myAtoi("00000-42a1234"))


# 实现 strStr
def get_next(needle):
    # 生成list
    next = [0] * len(needle)
    k = -1
    next[0] = k
    for i in range(1, len(needle)):
        # 第一个-1,后面有匹配的0到每个匹配,不匹配继续就-1
        while k > -1 and needle[i] != needle[k + 1]:
            # 不匹配就回退
            k = next[k]
        if needle[i] == needle[k + 1]:
            # 匹配+1
            k += 1
        next[i] = k
    return next


# KMP
def strStr(haystack, needle):
    a = len(needle)
    b = len(haystack)
    if a == 0:
        return 0
    next = get_next(needle)
    p = -1
    for j in range(b):
        # 没有匹配的就会退到上一个有匹配的
        while p >= 0 and needle[p + 1] != haystack[j]:
            p = next[p]
        if needle[p + 1] == haystack[j]:
            p += 1
        if p == a - 1:
            return j - a + 1
    return -1


haystack = "aabaabaaf"
needle = "aabaaf"

print(strStr(haystack, needle))


# 外观数列
def countAndSay(n):
    if n == 1:
        return '1'
    # 递归
    result1 = countAndSay(n - 1)
    n = str(result1)
    temp = [0, 0]
    result = ''
    for i in range(len(n)):
        if temp[1] == 0:
            temp[1] = 1
            temp[0] = n[i]
        elif temp[0] == n[i]:
            temp[1] += 1
        else:
            result += str(temp[1])
            result += temp[0]
            temp = [n[i], 1]
    result += str(temp[1])
    result += temp[0]
    return result


n = 5
print(countAndSay(n))


# 最长公共前缀
def longestCommonPrefix(strs):
    lenx = 0
    for i in range(len(strs)):
        if len(strs[i]) < len(strs[lenx]):
            lenx = i
    result = strs[lenx]
    for i in range(len(strs)):
        cut = 0
        while cut < len(result):
            if strs[i][cut] == result[cut]:
                cut += 1
            else:
                result = result[0:cut]
                break
    return result


strs = ["flower", "flow", "flight"]
print(longestCommonPrefix(strs))

'''

print(1)
