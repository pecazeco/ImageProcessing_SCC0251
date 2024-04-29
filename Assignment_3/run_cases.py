import os
import time

''' 
if you only want to run one of the cases:
    python main.py < test_cases_in_out/case1.in
    python main.py < test_cases_in_out/case2.in
    ...
'''

open('all_cases_output.txt', 'w').close()

num_cases = 12
ti_total = time.time()
    
for case in range(1,num_cases+1):
    
    print(f"Running case {case}", file=open("all_cases_output.txt", "a"))
    
    ti = time.time()
    
    os.system(f"python main.py < test_cases_in_out/case{case}.in >> all_cases_output.txt")
    
    print('Execution time [s]:', time.time()-ti, file=open("all_cases_output.txt", "a"))
    print('-----------------------------------', file=open("all_cases_output.txt", "a"))
    
print("Total execution time [s]:", time.time()-ti_total, file=open("all_cases_output.txt", "a"))
