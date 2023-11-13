lecture_list = ['YNB_4876', 'PLS_3824', 'YDT_4892', 'DXC_5781', 'NKT_3877', 'IBZ_8016', 'TPK_3513', 'HHZ_2871', 'KQP_1484', 'RKR_3989']

#TODO: FILL HERE!


lecture_code = []
lecture_num = []
n1 = 0
n2 = 0

# for i in range(len(lecture_list)):
#     lecture_code.append(lecture_list[i].split('_')[0])
#     lecture_num.append(lecture_list[i].split('_')[1])



n1 = input("Enter the Grade: ")
n2 = input("Enter the Grade: ")

for i in lecture_list:
    if(i[4] >= n1 and i[4] <= n2):
        print(i)
