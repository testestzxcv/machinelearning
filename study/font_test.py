from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import mglearn
import matplotlib.font_manager as fm

data = np.random.randint(-100, 100, 50).cumsum()
print(data)

plt.plot(range(50), data, 'r')
mpl.rcParams['axes.unicode_minus'] = False
plt.title('시간별 가격 추이')
plt.ylabel('주식 가격')
plt.xlabel('시간(분)')

print ('버전: ', mpl.__version__)
print ('설치 위치: ', mpl.__file__)
print ('설정 위치: ', mpl.get_configdir())
print ('캐시 위치: ', mpl.get_cachedir())

print ('설정 파일 위치: ', mpl.matplotlib_fname())

font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')

# ttf 폰트 전체개수
print(len(font_list))

font_list_mac = fm.OSXInstalledFonts()
print(len(font_list_mac))

print(font_list[:10])

f = [f.name for f in fm.fontManager.ttflist]
print(len(font_list))
# 10개의 폰트 명 만 출력
print(f[:10])

[(f.name, f.fname) for f in fm.fontManager.ttflist if 'Nanum' in f.name]

# 기본 설정 읽기
import matplotlib.pyplot as plt

# size, family
print('# 설정되어있는 폰트 사이즈')
print (plt.rcParams['font.size'] )
print('# 설정되어있는 폰트 글꼴')
print (plt.rcParams['font.family'] )

# serif, sans-serif, monospace
print('serif 세리프가 있는 폰트--------')
print (plt.rcParams['font.serif'])
print('sans-serif 세리프가 없는 폰트 --------')
print (plt.rcParams['font.sans-serif'])
print('monospace 고정폭 글꼴--------')
print (plt.rcParams['font.monospace'])

plt.rcParams["font.family"] = 'Nanum Brush Script OTF'
plt.rcParams["font.size"] = 20
plt.rcParams["figure.figsize"] = (14,4)

plt.plot(range(50), data, 'r')
plt.title('시간별 가격 추이')
plt.ylabel('주식 가격')
plt.xlabel('시간(분)')
plt.style.use('seaborn-pastel')
plt.show()

path = '/Library/Fonts/NanumBarunGothic.ttf'
font_name = fm.FontProperties(fname=path, size=50).get_name()
print(font_name)
plt.rc('font', family=font_name)

fig, ax = plt.subplots()
ax.plot(data)
ax.set_title('시간별 가격 추이')
plt.ylabel('주식 가격')
plt.xlabel('시간(분)')
plt.style.use('ggplot')
plt.show()

# # fname 옵션을 사용하는 방법
# path = '/Library/Fonts/NanumBarunpenRegular.otf'
# fontprop = fm.FontProperties(fname=path, size=18)
#
# plt.plot(range(50), data, 'r')
# plt.title('시간별 가격 추이', fontproperties=fontprop)
# plt.ylabel('주식 가격', fontproperties=fontprop)
# plt.xlabel('시간(분)', fontproperties=fontprop)
# plt.show()