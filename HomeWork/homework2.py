#!/user/bin/env python3
# -*- conding:utf-8 -*-
# @auther Zhang
# @date 2021/9/30
# @file homework2.py
import math

'''
十进制转换为其他进制：
    十进制转二进制  bin(10) -> '0b1010'
    十进制转八进制  oct(10)
    十进制转十六进制 hex(10)
方法二：format
'{:b}'.format(9)
'1001'
>>> '{:o}'.format(9)
'11'
>>> '{:x}'.format(10)
'a'

其他进制转十进制：
    int()函数
    int('0b1010',2) --> 10 或者 int('1010',2) --> 10  
    int('0o12',8)   --> 10 或者 int('12',8)   --> 10  
    int('0xa',16)   --> 10 或者 int('a',16)   --> 10
    备注：int(x,y)括号中x是需要转换的数值type必须是字符串,y是当前进制数

31 3.将十进制转换为固定长度的多进制类型：
32 方法一:
33 >>> '{:08b}'.format(9)
34 '00001001'
35 >>> '{:06o}'.format(9)
36 '000011'
37 >>> '{:06x}'.format(9)
38 '000009'

'''
# https://blog.csdn.net/Xavier_8031/article/details/103647786?utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7Eessearch%7Evector-13.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7Eessearch%7Evector-13.no_search_link

def Karatsuba(x, y):
    '''n位10进制大数相乘'''
    '''递归终止条件'''
    if len(str(x)) == 1 or len(str(y)) == 1:
        return x * y

    n = max(len(str(x)), len(str(y)))
    '''需要注意的是，此处的k必须是整数'''
    k = n // 2

    x1 = x // 10 ** k
    x0 = x % 10 ** k
    y1 = x // 10 ** k
    y0 = x % 10 ** k

    z0 = Karatsuba(x0, y0)
    z2 = Karatsuba(x1, y1)
    z1 = Karatsuba((x1 + x0), (y1 + y0)) - z2 - z0

    xy = z2 * 10 ** (k * 2) + z1 * 10 ** (k) + z0

    return xy


def main():
    option = input('是否选择使用默认数据（Y/N）: ')
    if option == 'Y':
        x = 767389078975544323523755
        y = 643289078976565442454599
    else:
        x = input('请输入x：')
        y = input('请输入y：')
    xy = Karatsuba(x, y)
    print(xy)


main()
