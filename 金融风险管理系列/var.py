import pandas as pd
import numpy as np

d1 = pd.read_excel('doge.xlsx').values
a1 = np.cov(np.transpose(d1),ddof=0, aweights=np.ones_like(d1[:, 0]) * 0.5)
print(a1)
#A.B可多次二分然后并行
A = d1[0:10]
B = d1[10:]
og_A = len(A) * 0.5
og_B = len(B) * 0.5
og_AB = og_A + og_B
V_A = np.sum(A[:, 0] * 0.5)
V_B = np.sum(B[:, 0] * 0.5)
W_A = np.sum(A[:, 1] * 0.5)
W_B = np.sum(B[:, 1] * 0.5)

VW_A = np.sum(0.5 * ((A[:, 0]) - (1 / og_A) * V_A) * ((A[:, 1]) - (1 / og_A) * W_A))
VW_B = np.sum(0.5 * ((B[:, 0]) - (1 / og_B) * V_B) * ((B[:, 1]) - (1 / og_B) * W_B))

vx_A = np.sum(A[:, 0] * 0.5) / og_A
vx_B = np.sum(B[:, 0] * 0.5) / og_B
wx_A = np.sum(A[:, 1] * 0.5) / og_A
wx_B = np.sum(B[:, 1] * 0.5) / og_B

b2m = np.mean(B, axis=0)
VW_AB1 = VW_A + VW_B + (og_A * og_B) / og_AB * (vx_A - vx_B) * (wx_A - wx_B)
VW_AB2 = VW_A + VW_B + (og_B * V_A - og_A * V_B) * (og_B * W_A - og_A * W_B) / (og_A * og_B * og_AB)
print(VW_AB1 / og_AB)
print(VW_AB2 / og_AB)
