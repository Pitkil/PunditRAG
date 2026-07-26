import copy

# 原始列表，包含一个不可变元素(整数)和一个可变元素(列表)
original_list = [1, [2, 3]]

# 1. 赋值 (引用传递)
assigned_list = original_list

# 2. 浅拷贝
shallow_copied_list = copy.copy(original_list)

# 3. 深拷贝
deep_copied_list = copy.deepcopy(original_list)

# ================= 测试修改 =================

# 修改嵌套的可变对象
original_list[1][0] = 'X' 

# 修改最外层的不可变对象
original_list[0] = 'A'

print("原始列表:", original_list)       
# 输出: ['A', ['X', 3]]

print("赋值列表:", assigned_list)       
# 输出: ['A', ['X', 3]] (完全跟着变)

print("浅拷贝列表:", shallow_copied_list) 
# 输出: [1, ['X', 3]] (外层独立，内层跟着变)

print("深拷贝列表:", deep_copied_list)   
# 输出: [1, [2, 3]] (完全独立，不受任何影响)