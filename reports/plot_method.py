"""Draw explanatory schematics, not simulator meshes or measured trajectories."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parent
FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams["svg.fonttype"] = "path"
plt.rcParams["axes.unicode_minus"] = False
INK, MUTED = "#152c43", "#526779"
COLORS = ["#087f8c", "#d58422", "#5861b5", "#bc4c72"]


def ant(ax, x, y, scale=1, color=INK):
    ax.add_patch(Ellipse((x, y), .62*scale, .23*scale, color=color))
    ax.add_patch(Circle((x+.36*scale, y+.015*scale), .09*scale, color=color))
    for dx, sign in [(-.22, -1), (-.1, 1), (.12, -1), (.23, 1)]:
        ax.plot([x+dx*scale, x+(dx+.13*sign)*scale, x+(dx+.3*sign)*scale],
                [y, y-.25*scale, y-.46*scale], color=color, lw=2)


def label(ax, x, y, text, size=11, **kwargs):
    ax.text(x, y, text, fontsize=size, color=INK, va="center", **kwargs)


def arrow(ax, start, end, color=INK):
    ax.annotate("", end, start, arrowprops={"arrowstyle": "->", "color": color, "lw": 2})


fig, axes = plt.subplots(2, 2, figsize=(15, 10.4))
fig.patch.set_facecolor("#f3f6fa")
fig.subplots_adjust(left=.035, right=.965, top=.87, bottom=.085, wspace=.07, hspace=.10)
fig.text(.035, .95, "Ant는 무엇을 바꾸며 배우는가?", fontsize=25, weight="bold", color=INK)
fig.text(.035, .905, "접촉할 지형 → 마찰 변화 → 성과에 맞는 난이도 → 짧은 시간 기억", fontsize=14, color=MUTED)
titles = ["01  Terrain DR | 발 디딜 곳을 바꾼다", "02  Friction DR | 같은 동작, 다른 미끄러짐",
          "03  Performance-based Curriculum | 잘 걸으면 더 어렵게", "04  4-frame History | 한 순간 대신 변화 과정을 본다"]
for ax, title, color in zip(axes.flat, titles, COLORS):
    ax.set(xlim=(0, 10), ylim=(0, 6))
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((.02,.02),9.96,5.96, boxstyle="round,pad=0,rounding_size=.2", fc="white", ec="#d8e1eb"))
    ax.add_patch(Rectangle((.25,5.07),.07,.56,color=color))
    label(ax,.5,5.35,title,13,weight="bold")

a=axes[0,0]
for i, (name, pct) in enumerate(zip(["Flat", "Noise", "Blocks", "Slope"], [20,30,30,20])):
    x0=.55+2.35*i
    x=np.linspace(x0,x0+1.9,80)
    if i==0: y=np.full_like(x,2.5)
    elif i==1: y=2.5+.09*np.sin(np.arange(80)*1.6)+.07*np.sin(np.arange(80)*.7)
    elif i==2: y=2.5+np.repeat([0,.20,.04,.30,.12,.24,.02,.15],10)
    else: y=2.5+.45*(1-np.abs(x-(x0+.95))/.95)
    a.fill_between(x,2.08,y,color=COLORS[0],alpha=.20)
    a.plot(x,y,color=COLORS[0],lw=2)
    ant(a,x0+.9,3.45,.95)
    label(a,x0+.95,1.72,f"{name} {pct}%",11,ha="center")
label(a,.55,4.42,"평지 · 요철 · 블록 · 경사에서 지지와 균형을 반복 학습",12)
label(a,.55,.83,"역할: 특정 바닥 형상에만 맞춘 보행을 줄이는 학습 분포",11)
label(a,.55,.40,"mesh seed 10000 · 타일 8 × 8 m · 높이 상한 설정 10 cm",10)

a=axes[0,1]
for x, low in [(2.5,True),(7.4,False)]:
    a.plot([x-1.55,x+1.55],[2.5,2.5],color=COLORS[1],lw=4)
    ant(a,x-.2,3.16,1.4)
    arrow(a,(x-.5,2.25),(x+(1.15 if low else -.05),2.25),COLORS[1])
    label(a,x,1.75,"낮은 μ: 더 쉽게 미끄러짐" if low else "높은 μ: 더 큰 접선 지지",11,ha="center")
label(a,.55,4.42,"reset마다 환경별 마찰 샘플 → 네 발에 동일한 재질 적용",12)
label(a,.55,.83,"역할: 접촉 힘이 달라도 전진 동작을 유지하도록 학습",11)
label(a,.55,.40,"Friction 단계 μs, μd: 0.4–1.6 · 초기 자세/속도 변화도 추가",10)

a=axes[1,0]
for i, (h, mu) in enumerate(zip([.4,.8,1.2,1.6],["0.7–1.3","0.5–1.5","0.4–1.6","0.3–1.7"])):
    x=.8+2.2*i
    a.add_patch(Rectangle((x,1.8),1.8,h,fc=COLORS[2],alpha=.15+.12*i))
    label(a,x+.9,1.48,mu,10,ha="center")
    if i<3: arrow(a,(x+1.0,2.1+h),(x+2.7,2.45+h),COLORS[2])
ant(a,1.6,2.75,.85)
label(a,.55,4.43,"episode의 +x 변위로 다음 지형 level과 마찰 범위 조정",12)
label(a,5.15,3.93,"> 3.0 m: level ↑",11)
label(a,5.15,3.54,"< 0.75 m: level ↓",11)
label(a,.55,.86,"역할: 쉬운 접촉에서 시작해 어려운 조합으로 학습 확대",11)
label(a,.55,.42,"시작 level 0–1 · 단계별 μ 범위 표시 · 학습 순서와 범위 동시 변경",10)

a=axes[1,1]
for i in range(4):
    x=.6+i*1.6
    a.add_patch(FancyBboxPatch((x,2.05),1.32,1.52,boxstyle="round,pad=.04",fc="#fbf1f5",ec=COLORS[3]))
    ant(a,x+.61,2.91+.06*np.sin(i),.78)
    label(a,x+.66,2.31,["t−3","t−2","t−1","t"][i],10,ha="center")
arrow(a,(6.97,2.8),(7.53,2.8),COLORS[3])
a.add_patch(FancyBboxPatch((7.65,2.05),1.8,1.52,boxstyle="round,pad=.04",fc="#f1f4f8",ec="#a2b1c1"))
label(a,8.55,2.96,"PPO",13,ha="center",weight="bold")
label(a,8.55,2.51,"8개 관절 행동",10,ha="center")
label(a,.55,4.43,"관절 · 속도 · 접촉 힘 · 이전 행동의 최근 관측을 연결",12)
label(a,3.65,1.56,"60차원 × 4 = 240차원 입력",11,ha="center")
label(a,.55,.86,"역할: 접촉 이후의 변화에서 미끄러짐·동역학 단서 제공",11)
label(a,.55,.42,"60 Hz · 첫–마지막 관측 간 약 50 ms · 명시적 마찰 추정기는 없음",10)
fig.text(.035,.035,"구현에 근거한 개념도입니다. 실제 mesh·미끄러짐 궤적·성능 측정값이 아닙니다.  |  누적 비교: PPO → Terrain → Friction → Curriculum → History",fontsize=10,color=MUTED)
fig.savefig(ROOT / "method_overview.png",dpi=180)
fig.savefig(ROOT / "method_overview.svg")
