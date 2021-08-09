"""
Visualiza the experiment results about QNN's running time (Figure 14 (a) in the paper)
"""
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np

t = OrderedDict()
t['QENN, NISQ'] = 240.08
t['QNNN, NISQ'] = 126.01
t['QENN, FT'] = 4.30
t['QNNN, FT'] = 3.00
t['MLP'] = 1/58

params={'font.family':'serif',
        'font.serif':'Times New Roman',
        'font.style':'normal',
        'font.weight':'normal', #or 'blod'
        }
from matplotlib import rcParams
rcParams.update(params)

y_pos = np.arange(0, 0.35*5, 0.35)
fig, ax = plt.subplots(figsize=(10.08, 4.0))
b = ax.barh(y_pos, list(t.values()), height=0.3)
ax.set_xlim(0, 300)
for i, rect in enumerate(b):
    w = rect.get_width()
    ax.text(w, rect.get_y()+rect.get_height()/2, str(round(list(t.values())[i], 2)), ha='left', va='center', fontsize=20)
plt.yticks(y_pos, list(t.keys()))
# plt.yticks(())
plt.xlabel('Running time: s/iter', fontsize=20)
plt.tick_params(labelsize=20)
plt.tight_layout()
plt.savefig('figure/runtime.pdf', dpi=600, format='pdf')
plt.show()