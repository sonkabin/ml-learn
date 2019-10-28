import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3) # (3,3)将整个窗口分为3行3列 (0, 0) 表示从0行0列开始做图
ax1.plot([1, 2], [1, 2])    # 画小图
ax1.set_title('ax1_title')  # 设置小图的标题

ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2) # colspan和rowspan默认跨度为1
ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
ax4 = plt.subplot2grid((3, 3), (2, 0))
ax5 = plt.subplot2grid((3, 3), (2, 1))

ax4.scatter([1, 2], [2, 2])
ax4.set_xlabel('ax4_x')
ax4.set_ylabel('ax4_y')

plt.show()

plt.figure()
gs = gridspec.GridSpec(3, 3)

ax6 = plt.subplot(gs[0, :])
ax7 = plt.subplot(gs[1, :2])
ax8 = plt.subplot(gs[1:, 2])
ax9 = plt.subplot(gs[-1, 0])
ax10 = plt.subplot(gs[-1, -2])

plt.show()