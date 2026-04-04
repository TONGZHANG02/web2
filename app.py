# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 19:37:54 2026

@author: zhang
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import sqlite3
from datetime import datetime
import threading

# ========== 配置中文字体（使用项目中的 simhei.ttf）==========
# 确保 simhei.ttf 文件与 app.py 在同一目录下
font_path = "simhei.ttf"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams['axes.unicode_minus'] = False
else:
    # 如果文件不存在，降级使用系统字体（仅用于本地测试）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False

# 导入 sgp 相关模块（用于 MPD/ANN/HSS 预测）
from sgp.io import (
    SGPIO, DatabaseIO, VariableIO, MPDIO, ANNIO, EquationIO,
    HSSIO, SoilParametersIO, HSSParametersIO, InputIO
)
from sgp.models import Model

# 页面配置 - 必须放在最前面
st.set_page_config(
    page_title="上海岩土工程参数智能分析平台",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="auto"
)

# ========== 统计功能（SQLite数据库版）==========
DB_FILE = "usage_stats.db"
db_lock = threading.Lock()

def init_db():
    """初始化数据库"""
    with db_lock:
        conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        c = conn.cursor()
        
        # 访问记录表（只记录首次访问）
        c.execute('''
            CREATE TABLE IF NOT EXISTS visits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user_ip TEXT
            )
        ''')
        
        # 计算记录表（只记录总次数）
        c.execute('''
            CREATE TABLE IF NOT EXISTS calculations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user_ip TEXT
            )
        ''')
        
        conn.commit()
        conn.close()

def record_visit_db():
    """记录网站访问（只统计首次进入）"""
    try:
        with db_lock:
            conn = sqlite3.connect(DB_FILE, check_same_thread=False)
            c = conn.cursor()
            c.execute("INSERT INTO visits (timestamp, user_ip) VALUES (?, ?)",
                      (datetime.now().isoformat(), "unknown"))
            conn.commit()
            conn.close()
    except:
        pass

def record_calculation_db():
    """记录计算次数（总次数）"""
    try:
        with db_lock:
            conn = sqlite3.connect(DB_FILE, check_same_thread=False)
            c = conn.cursor()
            c.execute("INSERT INTO calculations (timestamp, user_ip) VALUES (?, ?)",
                      (datetime.now().isoformat(), "unknown"))
            conn.commit()
            conn.close()
    except:
        pass

def get_stats_db():
    """从数据库获取统计数据"""
    try:
        with db_lock:
            conn = sqlite3.connect(DB_FILE, check_same_thread=False)
            c = conn.cursor()
            
            # 总访问量
            c.execute("SELECT COUNT(*) FROM visits")
            total_visits =  c.fetchone()[0]
            
            # 总计算次数
            c.execute("SELECT COUNT(*) FROM calculations")
            total_calculations = c.fetchone()[0]
            
            conn.close()
            
            return {
                "total_visits": total_visits,
                "total_calculations": total_calculations
            }
    except:
        return {
            "total_visits": 0,
            "total_calculations": 0
        }

# 初始化数据库
init_db()

# 自定义CSS样式 - 极致紧凑布局，消除顶部空白
st.markdown("""
<style>
    /* 全局重置边距，消除所有默认留白 */
    body, .stApp, .stAppViewContainer, .main {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* 主容器顶部零内边距 */
    .main .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    
    /* 隐藏顶部白条及菜单按钮 */
    header[data-testid="stHeader"] {
        display: none;
    }
    button[kind="header"] {
        display: none;
    }
    a[data-testid="stDeployButton"] {
        display: none;
    }
    
    .main { 
        background: linear-gradient(135deg, #f0f4ff 0%, #e0e7ff 100%);
    }
    
    /* 首页标题样式 - 上方无空白 */
    .hero-title {
        text-align: center;
        color: #1e3a8a;
        font-size: 2rem;
        font-weight: 700;
        margin-top: 0rem !important;
        margin-bottom: 0.1rem;
        padding-top: 0rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .hero-subtitle {
        text-align: center;
        color: #64748b;
        font-size: 0.95rem;
        margin-top: 0;
        margin-bottom: 0.6rem;
    }
    
    /* 功能卡片样式（紧凑） */
    .feature-card {
        background: white;
        border-radius: 20px;
        padding: 0.9rem 0.8rem;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.15);
        border: 2px solid transparent;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 20px 40px -5px rgba(59, 130, 246, 0.25);
        border-color: #3b82f6;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        transform: scaleX(0);
        transition: transform 0.4s ease;
    }
    
    .feature-card:hover::before {
        transform: scaleX(1);
    }
    
    .feature-icon {
        font-size: 2.2rem;
        margin-bottom: 0.3rem;
        display: block;
    }
    
    .feature-title {
        color: #1e3a8a;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    
    .feature-desc {
        color: #64748b;
        font-size: 0.75rem;
        line-height: 1.3;
    }
    
    /* 自定义按钮样式 - 蓝色（紧凑） */
    div.stButton > button {
        background-color: #3b82f6;
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.3rem 0.8rem;
        font-size: 0.8rem;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 2px 4px -1px rgba(59, 130, 246, 0.3);
    }
    div.stButton > button:hover {
        background-color: #2563eb;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.4);
    }
    
    /* 返回按钮样式 - 位置更靠上 */
    .back-button {
        position: fixed;
        top: 0.3rem;
        left: 1rem;
        z-index: 1000;
        background: linear-gradient(90deg, #3b82f6, #2563eb);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.4rem 1rem;
        font-size: 0.85rem;
        font-weight: 600;
        cursor: pointer;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.3);
        transition: all 0.3s;
    }
    
    .back-button:hover {
        transform: translateX(-3px);
        box-shadow: 0 6px 8px -1px rgba(37, 99, 235, 0.4);
    }
    
    /* 内容页面头部 - 更扁，减少上方留白 */
    .page-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 15px;
        margin-top: 0rem !important;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    
    .page-title {
        margin: 0;
        font-size: 1.4rem;
        font-weight: 700;
    }
    
    .page-subtitle {
        margin: 0.05rem 0 0 0;
        opacity: 0.9;
        font-size: 0.8rem;
    }
    
    /* 数据表格样式优化 - 无滚动条 */
    .stDataFrame {
        border-radius: 10px;
        overflow: visible !important;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
    .stDataFrame div[data-testid="stDataFrameResizable"] {
        overflow: visible !important;
    }
    
    /* 自定义HTML表格样式 */
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
        background: white;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
    
    .custom-table th {
        background-color: #f8fafc;
        color: #1e3a8a;
        font-weight: 600;
        padding: 12px;
        text-align: left;
        border-bottom: 2px solid #e2e8f0;
    }
    
    .custom-table td {
        padding: 10px 12px;
        border-bottom: 1px solid #e2e8f0;
        color: #334155;
    }
    
    .custom-table tr:hover {
        background-color: #f1f5f9;
    }
    
    .custom-table tr:last-child td {
        border-bottom: none;
    }
    
    /* 相关系数矩阵表格样式 */
    .corr-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
        background: white;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
    
    .corr-table th {
        background-color: #f8fafc;
        color: #1e3a8a;
        font-weight: 600;
        padding: 10px 8px;
        text-align: center;
        border-bottom: 2px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
    }
    
    .corr-table th:first-child {
        text-align: left;
        min-width: 100px;
    }
    
    .corr-table td {
        padding: 8px;
        text-align: center;
        border-bottom: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        color: #334155;
    }
    
    .corr-table td:first-child {
        text-align: left;
        font-weight: 500;
        background-color: #f8fafc;
        color: #1e3a8a;
    }
    
    .corr-table tr:hover td {
        background-color: #f1f5f9;
    }
    
    .corr-table tr:hover td:first-child {
        background-color: #e2e8f0;
    }
    
    .corr-table tr:last-child td {
        border-bottom: none;
    }
    
    /* 指标卡片 */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .metric-box {
        background: white;
        padding: 0.8rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid #3b82f6;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e3a8a;
    }
    
    .metric-label {
        color: #64748b;
        font-size: 0.75rem;
        margin-top: 0.2rem;
    }
    
    /* 加载动画 */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    /* 调整st.columns之间的间距，减少垂直空白 */
    .row-widget.stHorizontal {
        margin-bottom: 0.2rem;
    }
    
    /* 右上角统计面板样式 */
    .stats-corner {
        position: fixed;
        top: 0.5rem;
        right: 1rem;
        z-index: 1000;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        border-radius: 10px;
        padding: 0.5rem 0.8rem;
        color: white;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);
        display: flex;
        gap: 1rem;
        align-items: center;
    }
    
    .stats-corner-item {
        text-align: center;
        padding: 0 0.5rem;
        border-right: 1px solid rgba(255,255,255,0.3);
    }
    
    .stats-corner-item:last-child {
        border-right: none;
    }
    
    .stats-corner-number {
        font-size: 1.2rem;
        font-weight: 700;
        color: white;
        line-height: 1;
    }
    
    .stats-corner-label {
        font-size: 0.65rem;
        color: rgba(255,255,255,0.9);
        margin-top: 0.1rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------- 初始化 session state ----------
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'first_visit' not in st.session_state:
    st.session_state.first_visit = True

# 参数名称映射（带HTML下标格式，用于第二部分的表格）
PARAM_DISPLAY_NAMES = {
    "e": "e",
    "wl": "w<sub>l</sub>",
    "wp": "w<sub>p</sub>",
    "ccq": "c<sub>cq</sub>",
    "phicq": "φ<sub>cq</sub>",
    "Es": "E<sub>s</sub>"
}

# 参数名称映射（LaTeX格式，用于图表标题和轴标签）
PARAM_LATEX_NAMES = {
    "e": "e",
    "wl": "$w_{l}$",
    "wp": "$w_{p}$",
    "ccq": "$c_{cq}$",
    "phicq": "$\\varphi_{cq}$",
    "Es": "$E_{s}$"
}

# 参数名称映射（带_下标格式，用于第三部分的选择框）
PARAM_UNDERSCORE_NAMES = {
    "e": "e",
    "wl": "w_l",
    "wp": "w_p",
    "ccq": "c_cq",
    "phicq": "φ_cq",
    "Es": "E_s"
}

# 辅助函数：获取HTML显示名称（用于第二部分的表格）
def get_display_name(param):
    return PARAM_DISPLAY_NAMES.get(param, param)

# 辅助函数：获取_下标显示名称（用于第三部分的选择框和标签）
def get_underscore_name(param):
    return PARAM_UNDERSCORE_NAMES.get(param, param)

# 辅助函数：获取LaTeX名称（用于图表）
def get_latex_name(param):
    return PARAM_LATEX_NAMES.get(param, param)

# 辅助函数：将DataFrame转换为HTML表格（普通表格）
def df_to_html_table(df, escape_html=False):
    """将DataFrame转换为HTML表格"""
    html = '<table class="custom-table">'
    
    # 表头
    html += '<thead><tr>'
    for col in df.columns:
        html += f'<th>{col}</th>'
    html += '</tr></thead>'
    
    # 表体
    html += '<tbody>'
    for _, row in df.iterrows():
        html += '<tr>'
        for val in row:
            # 如果不需要转义，且值是字符串，直接插入
            if escape_html:
                import html as html_module
                val = html_module.escape(str(val))
            html += f'<td>{val}</td>'
        html += '</tr>'
    html += '</tbody>'
    html += '</table>'
    
    return html

# 辅助函数：将相关系数矩阵转换为HTML表格（带行列标题）
def corr_matrix_to_html_table(corr_matrix, row_names, col_names):
    """将相关系数矩阵转换为HTML表格，包含行列标题"""
    html = '<table class="corr-table">'
    
    # 表头（列标题）
    html += '<thead><tr><th></th>'  # 左上角空白单元格
    for col_name in col_names:
        html += f'<th>{col_name}</th>'
    html += '</tr></thead>'
    
    # 表体
    html += '<tbody>'
    for i, row_name in enumerate(row_names):
        html += '<tr>'
        html += f'<td>{row_name}</td>'  # 行标题
        for j, val in enumerate(corr_matrix[i]):
            html += f'<td>{val:.4f}</td>'
        html += '</tr>'
    html += '</tbody>'
    html += '</table>'
    
    return html

# ---------- 加载 sgp 模型（缓存，无随机种子）----------
@st.cache_resource(show_spinner=False)
def load_sgp_model():
    """加载并初始化 sgp 模型（包含 MPD、ANN、HSS），完全随机自举"""
    # 读取数据
    df = pd.read_csv("shanghai.csv")
    required = ["e", "wl", "wp", "ccq", "phicq", "Es"]
    df = df[required].dropna().copy()

    # MPD/ANN 输入变量
    inputs_dict = {
        "e": VariableIO(name="e", unit="", decimals=3),
        "wl": VariableIO(name="wl", unit="%", decimals=1),
        "wp": VariableIO(name="wp", unit="%", decimals=1),
        "ccq": VariableIO(name="ccq", unit="kPa", decimals=1),
        "phicq": VariableIO(name="phicq", unit="°", decimals=1),
        "Es": VariableIO(name="Es", unit="MPa", decimals=2),
    }

    # ANN 方程
    ann_equations = {
        "ccq": EquationIO(
            inputs=["e", "wl", "wp"],
            equation="9+48*tanh(0.53*tanh(-13.53*e+0.1*wl-0.17*wp+9.97)-0.07*tanh(0.84*e-0.08*wl+0.01*wp+1.9)-0.3*tanh(4.16*e+0.07*wl-0.07*wp-4.31)+0.88*tanh(2.63*e+0.2*wl-0.15*wp-4.84)-0.01)"
        ),
        "phicq": EquationIO(
            inputs=["e", "wl", "wp"],
            equation="10+18*tanh(-0.83*tanh(-1.35*e-0.13*wl+0.15*wp+1.67)-0.34*tanh(-0.51*e+3*wl-2.92*wp-42.96)+0.28*tanh(-3.05*e+0.03*wl+0.03*wp+1.04)+0.15*tanh(0.84*e-1.42*wl+1.36*wp+24.41)+0.02)"
        ),
        "Es": EquationIO(
            inputs=["e", "wl", "wp"],
            equation="1.6+9*tanh(-0.29*tanh(10.11*e-0.14*wl+0.07*wp-3.79)-0.17*tanh(4.67*e+0.32*wl-0.34*wp-8.36)+0.8*tanh(-0.9*e-0.01*wl+0.02*wp+1.91)+0.03)"
        ),
    }

    mpd_io = MPDIO(
        inputs=["log(e)", "log(wl)", "log(wp)", "log(ccq)", "log(phicq)", "log(Es)"],
        tolerance=0.001,
        bootstraps=1000,
        optimizer="Scipy-LBFGSB",
        optimizer_options={"maxiter": 100, "maxfun": 1000},
    )

    # ---------- HSS 模型参数定义（严格依据 shanghai-generate.py）----------
    hss_inputs = [
        InputIO(name="e", default=0.7, minimum=0.4, maximum=1.6, decimals=3, singleStep=0.01, unit=""),
        InputIO(name="Es", default=10, minimum=1.5, maximum=25, decimals=2, singleStep=0.5, unit="MPa"),
        InputIO(name="sigma", default=10, minimum=0, maximum=500, decimals=2, singleStep=1.0, unit="kPa"),
        InputIO(name="ps", default=10, minimum=0.001, maximum=30, decimals=1, singleStep=1.0, unit="MPa"),
    ]

    hss_outputs = [
        VariableIO(name="Eoed", unit="MPa", decimals=2),
        VariableIO(name="E50", unit="MPa", decimals=2),
        VariableIO(name="Eur", unit="MPa", decimals=2),
        VariableIO(name="phi", unit="°", decimals=1),
        VariableIO(name="c", unit="kPa", decimals=1),
        VariableIO(name="psi", unit="°", decimals=1),
        VariableIO(name="Rf", unit="", decimals=2),
        VariableIO(name="K0", unit="", decimals=2),
        VariableIO(name="G0", unit="MPa", decimals=1),
        VariableIO(name="m", unit="", decimals=2),
        VariableIO(name="gamma07", unit="", decimals="scientific"),
        VariableIO(name="nu", unit="", decimals=1),
        VariableIO(name="pref", unit="kPa", decimals=1),
    ]

    # 公共更新（依赖 e, Es）
    base_updates = [
        HSSParametersIO(
            inputs=["e", "Es"],
            outputs=dict(
                Eoed="1.1+5*tanh(-0.01*tanh(0.04*e-0.02*Es+0.03)+1.49*tanh(1.29*e+0.41*Es-4)+1.9*tanh(-0.1*e+0.09*Es+1.59)-0.41*tanh(2.13*e+0.1*Es-1.72)+0.02)",
                E50="1.6+5.3*tanh(-0.27*tanh(-0.33*e+0.08*Es+0.15)+2.81*tanh(-3.44*e-0.06*Es+2.79)-1.88*tanh(-2.42*e-0.56*Es+3.13)+1.2*tanh(1.81*e+0.34*Es-1.47)-0.008)",
                Eur="10.4+32*tanh(1.03*tanh(2.94*e-1.33*Es+0.56)+1.2*tanh(-6.15*e+0.94*Es+4.21)+0.32*tanh(-0.28*e-0.19*Es+1.42)+0.62*tanh(0.14*e+0.23*Es-0.28)-0.02)",
            )
        )
    ]

    # 黏土参数
    clay_base = [
        HSSParametersIO(
            inputs=["e"],
            outputs=dict(
                Eoed="-4.34*ln(e)+3.51",
                E50="-5.33*ln(e)+3.95",
                Eur="-28.8*ln(e)+24.5",
                phi="30.4*e**(-0.65)",
                c="-0.76*(30.4*e**(-0.65)-38)",
                psi="0",
                Rf="0.95-0.9*min(max(e-1,0),0.5)",
                K0="0.95-sin(30.4*e**(-0.65)*pi/180)",
                G0="67.5*e**(-1.57)",
                m="0.65",
                gamma07="3.2*10**(-4)",
                nu="0.2",
                pref="100.0",
            )
        )
    ]

    # 黏土更新
    clay_updates = base_updates + [
        HSSParametersIO(
            inputs=["e", "sigma", "ps"],
            outputs=dict(
                G0="(16.155+146.112*tanh(-0.331*tanh(0.441*e-0.007*sigma+1.066)+0.722*tanh(-2.681*e+0.027*sigma-4.991)+1.205*tanh(-0.34*e+0.00004*sigma+1.792)-0.004*tanh(-0.076*e-0.00023*sigma+0.12))) / (((0.95-sin(0.856*(13.56*ps**(-0.7)+atan(0.126*ln(ps)+0.24)*180/pi)*pi/180))*sigma/100.0)**0.65)"
            ),
        ),
        HSSParametersIO(
            inputs=["e", "ps"],
            outputs=dict(
                phi="0.856*(13.56*ps**(-0.7)+atan(0.126*ln(ps)+0.24)*180/pi)",
                c="-0.76*(0.856*(13.56*ps**(-0.7)+atan(0.126*ln(ps)+0.24)*180/pi)-38)",
                K0="0.95-sin(0.856*(13.56*ps**(-0.7)+atan(0.126*ln(ps)+0.24)*180/pi)*pi/180)",
            ),
        ),
    ]

    # 砂土基础参数
    sand_base = [
        HSSParametersIO(
            inputs=["e"],
            outputs=dict(
                Eoed="0.81*111*exp(-2.89*e)",
                E50="1.02*111*exp(-2.89*e)",
                Eur="4.2*111*exp(-2.89*e)+7.24",
                phi="26.9*e**(-0.72)",
                c="-1.32*(26.9*e**(-0.72)-33)",
                Rf="0.95-0.9*min(max(e-1,0),0.5)",
                psi="max(26.9*e**(-0.72)-30,0)",
                K0="1-sin(26.9*e**(-0.72)*pi/180)",
                G0="98.9*e**(-0.45)",
                m="0.7",
                gamma07="3.9*10**(-4)",
            )
        )
    ]

    # 砂土更新
    sand_updates = base_updates + [
        HSSParametersIO(
            inputs=["e", "sigma", "ps"],
            outputs=dict(
                G0="(16.155+146.112*tanh(-0.331*tanh(0.441*e-0.007*sigma+1.066)+0.722*tanh(-2.681*e+0.027*sigma-4.991)+1.205*tanh(-0.34*e+0.00004*sigma+1.792)-0.004*tanh(-0.076*e-0.00023*sigma+0.12))) / (((1-sin(0.856*(13.56*ps**(-0.7)+atan(0.126*ln(ps)+0.24)*180/pi)*pi/180))*sigma/100.0)**0.7)"
            ),
        ),
        HSSParametersIO(
            inputs=["e", "ps"],
            outputs=dict(
                phi="0.856*(13.56*ps**(-0.7)+atan(0.126*ln(ps)+0.24)*180/pi)",
                c="-1.32*(0.856*(13.56*ps**(-0.7)+atan(0.126*ln(ps)+0.24)*180/pi)-33)",
                psi="max(0.856*(13.56*ps**(-0.7)+atan(0.126*ln(ps)+0.24)*180/pi)-30,0)",
                K0="1-sin(0.856*(13.56*ps**(-0.7)+atan(0.126*ln(ps)+0.24)*180/pi)*pi/180)",
            ),
        ),
    ]

    hss_io = HSSIO(
        inputs=hss_inputs,
        outputs=hss_outputs,
        parameters={
            "黏土": SoilParametersIO(parameters=clay_base, updates=clay_updates),
            "砂土": SoilParametersIO(parameters=sand_base, updates=sand_updates),
        },
    )

    # 构建 DatabaseIO
    database = DatabaseIO(
        inputs=inputs_dict,
        data=df.values.tolist(),
        stats={},
        ann=ANNIO(equations=ann_equations),
        mpd=mpd_io,
        hss=hss_io,
    )

    sgp_io = SGPIO(database=database)
    model = Model(sgp_io)
    _ = model.mpd.dist()  # 预计算 MPD 分布（自举随机）

    return model

# ---------- 导航函数 ----------
def go_home():
    st.session_state.current_page = 'home'
    st.rerun()

def go_page(page_name):
    st.session_state.current_page = page_name
    st.rerun()

# ---------- 数据加载函数（用于首页）----------
def load_data():
    """智能加载数据"""
    local_file = "./shanghai-stats.csv"
    
    if os.path.exists(local_file):
        try:
            df = pd.read_csv(local_file)
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.session_state.data_source = f"本地文件: {local_file}"
            return df
        except Exception as e:
            st.error(f"读取本地文件失败: {e}")
    return None

# ==================== 页面路由 ====================
if st.session_state.current_page == 'home':
    # 记录首次访问（只记录一次）
    if st.session_state.first_visit:
        record_visit_db()
        st.session_state.first_visit = False
    
    # 获取统计数据
    stats = get_stats_db()
    
    # 首页
    st.markdown('<h1 class="hero-title">🏗️ 上海岩土工程参数智能分析平台</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">基于上海地区工程地质数据的专业分析工具</p>', unsafe_allow_html=True)
    
    # 右上角统计面板
    st.markdown(f"""
    <div class="stats-corner">
        <div class="stats-corner-item">
            <div class="stats-corner-number">{stats["total_visits"]}</div>
            <div class="stats-corner-label">总访问量</div>
        </div>
        <div class="stats-corner-item">
            <div class="stats-corner-number">{stats["total_calculations"]}</div>
            <div class="stats-corner-label">总计算次数</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        load_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown("""
            <div class="feature-card">
                <span class="feature-icon">📊</span>
                <div class="feature-title">上海规范推荐地层参数</div>
                <div class="feature-desc">查看不同地层类型的物理力学参数统计分布</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("进入参数分布", key="btn_dist", use_container_width=True):
                go_page('distribution')
        
        with st.container():
            st.markdown("""
            <div class="feature-card" style="margin-top: 0.5rem;">
                <span class="feature-icon">🔮</span>
                <div class="feature-title">土体参数智能预测</div>
                <div class="feature-desc">基于多元概率分布模型与人工神经网络模型的土体参数预测</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("进入参数预测", key="btn_pred", use_container_width=True):
                go_page('prediction')
    
    with col2:
        with st.container():
            st.markdown("""
            <div class="feature-card">
                <span class="feature-icon">📈</span>
                <div class="feature-title">多元概率分布分析</div>
                <div class="feature-desc">查看参数统计、Johnson分布参数及对数相关系数矩阵</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("进入多元分析", key="btn_multi", use_container_width=True):
                go_page('multivariate')
        
        with st.container():
            st.markdown("""
            <div class="feature-card" style="margin-top: 0.5rem;">
                <span class="feature-icon">🧱</span>
                <div class="feature-title">HSS模型参数计算</div>
                <div class="feature-desc">小应变硬化模型参数经验计算</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("进入HSS模型", key="btn_hss", use_container_width=True):
                go_page('hss')

# ==================== 参数分布页面 ====================
elif st.session_state.current_page == 'distribution':
    if st.button("← 返回首页", key="back_dist"):
        go_home()
    
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">📊 上海规范推荐地层参数分布</h1>
        <p class="page-subtitle">基于《上海市地基基础设计标准》（DGJ08-11-2018）的地层参数统计分析</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.error("请先返回首页加载数据")
        st.stop()
    
    df = st.session_state.df
    st.subheader("📋 地层参数数据表")
    st.dataframe(df, use_container_width=True, height=600)
    
    # 添加总表下载按钮
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 下载完整数据表",
        data=csv,
        file_name="shanghai-stats.csv",
        mime="text/csv"
    )
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("记录总数", len(df))
        with col2:
            st.metric("数值参数", len(numeric_cols))
        with col3:
            st.metric("参数均值", f"{df[numeric_cols[0]].mean():.2f}" if numeric_cols else "N/A")
        with col4:
            st.metric("标准差", f"{df[numeric_cols[0]].std():.2f}" if numeric_cols else "N/A")
        
        st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)

# ==================== 多元概率分布分析页面 ====================
elif st.session_state.current_page == 'multivariate':
    if st.button("← 返回首页", key="back_multi"):
        go_home()
    
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">📈 多元概率分布分析</h1>
        <p class="page-subtitle">参数统计、Johnson分布参数及对数相关系数矩阵</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 确保模型已加载
    if 'model' not in st.session_state:
        with st.spinner("正在加载..."):
            st.session_state.model = load_sgp_model()
    model = st.session_state.model
    
    # 从原始数据中提取六个参数
    params = ["e", "wl", "wp", "ccq", "phicq", "Es"]
    df_raw = pd.read_csv("shanghai.csv")
    df = df_raw[params].dropna()
    
    # ---------- 1. 参数统计表 ----------
    st.subheader("📊 参数统计表")
    stats_data = []
    for col in params:
        mean_val = df[col].mean()
        min_val = df[col].min()
        max_val = df[col].max()
        std_val = df[col].std()
        cov_val = std_val / mean_val if mean_val != 0 else np.nan
        stats_data.append({
            "参数": get_display_name(col),
            "均值": f"{mean_val:.4f}",
            "最小值": f"{min_val:.4f}",
            "最大值": f"{max_val:.4f}",
            "标准差": f"{std_val:.4f}",
            "变异系数": f"{cov_val:.4f}"
        })
    df_stats = pd.DataFrame(stats_data)
    # 使用HTML表格显示
    st.markdown(df_to_html_table(df_stats), unsafe_allow_html=True)
    
    # ---------- 2. 约翰逊分布参数表 ----------
    st.subheader("📈 约翰逊分布参数（对数空间）")
    # 获取分布结果
    dist_result = model.mpd.dist()
    
    # 获取每个变量的最优 z 值（通过重新调用 optimize，利用缓存）
    log_vars = ["log(e)", "log(wl)", "log(wp)", "log(ccq)", "log(phicq)", "log(Es)"]
    df_log = pd.DataFrame()
    for var, log_var in zip(params, log_vars):
        df_log[log_var] = np.log(df[var])
    
    # 调用 optimize 获取优化结果（包含 z 值）
    with st.spinner("正在加载..."):
        # 注意 optimize 返回的是 (results, statistics, pvalues) 元组
        opt_results, _, _ = model.mpd.optimize(
            df_log,
            method=model.io.database.mpd.optimizer,
            **model.io.database.mpd.optimizer_options
        )
    
    johnson_data = []
    for var, log_var in zip(params, log_vars):
        # 从 dist_result 中获取分布参数
        d = dist_result.dists[var]  # 注意 dist_result.dists 的键是原始变量名
        # 从 opt_results 中获取 z 值（opt_results 是字典，键为 log_var）
        z_val = opt_results[log_var].x[0]
        # 构建对数变量显示名称
        if var == "e":
            log_display = "log(e)"
        elif var == "wl":
            log_display = "log(w<sub>l</sub>)"
        elif var == "wp":
            log_display = "log(w<sub>p</sub>)"
        elif var == "ccq":
            log_display = "log(c<sub>cq</sub>)"
        elif var == "phicq":
            log_display = "log(φ<sub>cq</sub>)"
        elif var == "Es":
            log_display = "log(E<sub>s</sub>)"
        else:
            log_display = log_var
            
        johnson_data.append({
            "参数": log_display,
            "类型": d.type.upper(),
            "aX": f"{d.aX:.4f}",
            "bX": f"{d.bX:.4f}",
            "aY": f"{d.aY:.4f}",
            "bY": f"{d.bY:.4f}",
            "z": f"{z_val:.4f}",
            "D": f"{d.statistic:.4f}",
            "p": f"{d.pvalue:.4f}"
        })
    df_johnson = pd.DataFrame(johnson_data)
    # 使用HTML表格显示
    st.markdown(df_to_html_table(df_johnson), unsafe_allow_html=True)
    
    # ---------- 3. 对数相关系数表 ----------
    st.subheader("🔗 对数相关系数矩阵")
    # 使用自举计算的相关系数矩阵（X 空间，但对应对数变量）
    corr_matrix = dist_result.C  # 自举后的相关系数矩阵
    
    # 准备行列标题（带HTML下标）
    log_vars_display = ["log(e)", "log(w<sub>l</sub>)", "log(w<sub>p</sub>)", "log(c<sub>cq</sub>)", "log(φ<sub>cq</sub>)", "log(E<sub>s</sub>)"]
    
    # 使用专门的相关系数矩阵HTML表格函数
    st.markdown(corr_matrix_to_html_table(corr_matrix, log_vars_display, log_vars_display), unsafe_allow_html=True)

# ==================== 土体参数智能预测页面（MPD + ANN）====================
elif st.session_state.current_page == 'prediction':
    if st.button("← 返回首页", key="back_pred"):
        go_home()
    
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">🔮 土体参数智能预测</h1>
        <p class="page-subtitle">支持多元概率分布预测 (MPD) 和 ANN预测</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 延迟加载模型
    if 'model' not in st.session_state:
        with st.spinner("正在加载..."):
            st.session_state.model = load_sgp_model()
    model = st.session_state.model
    mpd = model.mpd
    
    # 数据统计（仅用于默认值）
    df_raw = pd.read_csv("shanghai.csv")
    original_vars = ["e", "wl", "wp", "ccq", "phicq", "Es"]
    df = df_raw[original_vars].dropna()
    sample_means = df.mean().to_dict()
    
    # 参数中英文对照（使用_下标，用于第三部分的选择框和标签，但ANN模式下我们会用HTML标签覆盖）
    param_names_zh = {
        "e": "孔隙比 e",
        "wl": "液限 w_l (%)",
        "wp": "塑限 w_p (%)",
        "ccq": "固结快剪黏聚力 c_cq (kPa)",
        "phicq": "固结快剪内摩擦角 φ_cq (°)",
        "Es": "压缩模量 E_s (MPa)"
    }
    
    # 创建两列布局：左侧参数输入，右侧结果展示
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.subheader("⚙️ 参数设置")
        mode = st.radio(
            "选择预测模式",
            ["多元概率分布预测 (MPD)", "ANN预测"],
            index=0
        )
        
        if mode == "ANN预测":
            # ANN预测模式：固定输入参数为 e, wl, wp，使用HTML标签显示下标
            st.info("ANN预测需要输入孔隙比 e、液限 w_l、塑限 w_p")
            input_vars = ["e", "wl", "wp"]
            input_values = {}
            # 使用自定义布局显示带下标的标签
            with st.container():
                # 孔隙比 e
                e_val = st.number_input("孔隙比 e", value=float(sample_means["e"]), format="%.3f", key="ann_e")
                input_values["e"] = e_val
                # 液限 w_l
                wl_val = st.number_input("液限 w_l (%)", value=float(sample_means["wl"]), format="%.3f", key="ann_wl")
                input_values["wl"] = wl_val
                # 塑限 w_p
                wp_val = st.number_input("塑限 w_p (%)", value=float(sample_means["wp"]), format="%.3f", key="ann_wp")
                input_values["wp"] = wp_val
            target_var = None
        else:
            # MPD 模式：用户自由选择已知参数
            input_vars = st.multiselect(
                "已知参数", 
                original_vars, 
                default=[],
                format_func=lambda x: param_names_zh[x]
            )
            input_values = {}
            for var in input_vars:
                mean_val = sample_means[var]
                val = st.number_input(
                    param_names_zh[var], value=float(mean_val), format="%.3f", key=f"mpd_{var}"
                )
                input_values[var] = val
            
            # 目标参数选项：排除已选中的已知参数
            target_options = [v for v in original_vars if v not in input_vars]
            if not target_options:
                st.warning("请至少保留一个参数作为目标参数")
                target_var = None
            else:
                target_var = st.selectbox(
                    "目标参数", 
                    target_options, 
                    index=0,
                    format_func=lambda x: param_names_zh[x]
                )
        
        calc_btn = st.button("🚀 计算", type="primary", use_container_width=True)
    
    with col_right:
        if calc_btn:
            if mode == "多元概率分布预测 (MPD)" and target_var is not None:
                # 记录计算
                record_calculation_db()
                
                with st.spinner("正在加载..."):
                    result_dict = mpd.predict(**input_values)
                    res = result_dict[target_var]

                # 提取数据
                x_uncond = np.asarray(res.unconditioning.y0)
                pdf_uncond = np.asarray(res.unconditioning.pdf)
                lb_uncond = res.unconditioning.lb
                ub_uncond = res.unconditioning.ub
                mean_uncond_display = res.unconditioning.mean
                pdf_at_mean_uncond = np.interp(mean_uncond_display, x_uncond, pdf_uncond)

                x_cond = np.asarray(res.conditioning.y0)
                pdf_cond = np.asarray(res.conditioning.pdf)
                lb_cond = res.conditioning.lb
                ub_cond = res.conditioning.ub
                mean_cond_display = res.conditioning.mean
                pdf_at_mean_cond = np.interp(mean_cond_display, x_cond, pdf_cond)

                # 注意：全局字体已在文件开头配置，这里无需再设置
                # 获取目标参数的 LaTeX 格式（用于图表标题和轴标签）和 HTML 格式（用于提示）
                target_latex = get_latex_name(target_var)
                target_html = get_display_name(target_var)

                # 在图片上方显示目标参数（大字体，带渐变背景）
                st.markdown(f"""
                <div style="
                    background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
                    color: white;
                    padding: 12px 20px;
                    border-radius: 10px;
                    margin-bottom: 15px;
                    text-align: center;
                    font-size: 1.3rem;
                    font-weight: 600;
                    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
                    🎯 目标参数：{target_html}
                </div>
                """, unsafe_allow_html=True)

                # 绘图（使用 LaTeX 数学模式）
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(x_uncond, pdf_uncond, 'k--', label="无条件 PDF")
                mask_u = (x_uncond >= lb_uncond) & (x_uncond <= ub_uncond)
                ax.fill_between(x_uncond[mask_u], 0, pdf_uncond[mask_u], color='k', alpha=0.1,
                                label=f"无条件 95% CI (lb = {lb_uncond:.2f}, ub = {ub_uncond:.2f}, mean = {mean_uncond_display:.2f})")
                ax.plot([mean_uncond_display, mean_uncond_display], [0, pdf_at_mean_uncond],
                        'k--', alpha=0.5, label=f"无条件均值 = {mean_uncond_display:.2f}")

                ax.plot(x_cond, pdf_cond, 'r-', label="条件 PDF")
                mask_c = (x_cond >= lb_cond) & (x_cond <= ub_cond)
                ax.fill_between(x_cond[mask_c], 0, pdf_cond[mask_c], color='r', alpha=0.1,
                                label=f"条件 95% CI (lb = {lb_cond:.2f}, ub = {ub_cond:.2f}, mean = {mean_cond_display:.2f})")
                ax.plot([mean_cond_display, mean_cond_display], [0, pdf_at_mean_cond],
                        'r-', alpha=0.5, label=f"有条件均值 = {mean_cond_display:.2f}")

                units = {"e": "", "wl": "%", "wp": "%", "ccq": "kPa", "phicq": "°", "Es": "MPa"}
                ax.set_xlabel(f"{target_latex} ({units[target_var]})")
                ax.set_ylabel("概率密度")
                ax.set_title(f"{target_latex} 的概率密度函数 (PDF)")
                ax.legend(loc="upper right")
                ax.grid(linestyle=':', alpha=0.6)

                st.pyplot(fig)

                # 结果显示区域 - 使用带背景色的框美化
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    <div style="
                        background-color: #f3f4f6;
                        border-left: 4px solid #6b7280;
                        padding: 15px;
                        border-radius: 8px;
                        margin-top: 10px;">
                        <h4 style="margin: 0 0 10px 0; color: #374151; font-size: 1.1rem;">📊 无条件分布</h4>
                        <p style="margin: 5px 0; color: #4b5563;"><strong>均值:</strong> {:.2f}</p>
                        <p style="margin: 5px 0; color: #4b5563;"><strong>95% 置信区间:</strong> [{:.2f}, {:.2f}]</p>
                    </div>
                    """.format(mean_uncond_display, lb_uncond, ub_uncond), unsafe_allow_html=True)

                with col2:
                    if input_values:
                        cond_items = []
                        for k, v in input_values.items():
                            cond_items.append(f"{get_display_name(k)}={v:.3f}")
                        cond_str = ", ".join(cond_items)
                    else:
                        cond_str = "无"

                    st.markdown("""
                    <div style="
                        background-color: #fef2f2;
                        border-left: 4px solid #ef4444;
                        padding: 15px;
                        border-radius: 8px;
                        margin-top: 10px;">
                        <h4 style="margin: 0 0 10px 0; color: #991b1b; font-size: 1.1rem;">🎯 条件分布</h4>
                        <p style="margin: 5px 0; color: #7f1d1d;"><strong>条件:</strong> {}</p>
                        <p style="margin: 5px 0; color: #7f1d1d;"><strong>均值:</strong> {:.2f}</p>
                        <p style="margin: 5px 0; color: #7f1d1d;"><strong>95% 置信区间:</strong> [{:.2f}, {:.2f}]</p>
                    </div>
                    """.format(cond_str, mean_cond_display, lb_cond, ub_cond), unsafe_allow_html=True)
            
            elif mode == "多元概率分布预测 (MPD)" and target_var is None:
                st.error("无法预测：请至少保留一个参数作为目标参数")
            
            else:  # ANN预测
                # 记录计算
                record_calculation_db()
                
                with st.spinner("正在加载..."):
                    try:
                        ann_results = model.ann.predict(**input_values)
                    except Exception as e:
                        st.error(f"ANN 预测失败: {e}")
                        ann_results = {}
                
                if ann_results:
                    st.subheader("🧠 ANN 预测结果")
                    # 构建带下标的显示数据，预测值保留两位小数
                    ann_data = []
                    for param, value in ann_results.items():
                        units_map = {"ccq": "kPa", "phicq": "°", "Es": "MPa"}
                        ann_data.append({
                            "参数": get_display_name(param),
                            "预测值": f"{value:.2f}",
                            "单位": units_map.get(param, "")
                        })
                    df_ann = pd.DataFrame(ann_data)
                    # 使用HTML表格显示
                    st.markdown(df_to_html_table(df_ann), unsafe_allow_html=True)
                else:
                    st.warning("未能获取 ANN 预测结果，请检查输入参数是否包含 e、w_l、w_p（ANN 的输入要求）")
        else:
            st.info("👈 请在左侧设置参数后点击「计算」")

# ==================== HSS模型参数计算页面 ====================
elif st.session_state.current_page == 'hss':
    if st.button("← 返回首页", key="back_hss"):
        go_home()
    
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">🧱 HSS模型参数计算</h1>
        <p class="page-subtitle">小应变硬化模型参数计算（基于上海地区经验公式）</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 延迟加载模型
    if 'model' not in st.session_state:
        with st.spinner("正在加载..."):
            st.session_state.model = load_sgp_model()
    model = st.session_state.model
    hss = model.io.database.hss
    
    # 参数含义字典（更新为带下标的名称）
    param_meaning = {
        "Eoed": "切线模量（参考应力下）",
        "E50": "割线模量（参考应力下）",
        "Eur": "加卸载模量（参考应力下）",
        "phi": "有效内摩擦角",
        "c": "有效粘聚力",
        "psi": "剪胀角",
        "Rf": "破坏比",
        "K0": "初始静止侧压力系数",
        "G0": "动剪切初始模量（参考应力下）",
        "m": "模量应力水平相关幂指数",
        "gamma07": "阈值剪应变",
        "nu": "泊松比",
        "pref": "参考应力",
    }
    
    # 参数显示名称映射（HTML下标格式）
    param_display_names = {
        "Eoed": "E<sub>oed</sub>",
        "E50": "E<sub>50</sub>",
        "Eur": "E<sub>ur</sub>",
        "phi": "φ",
        "c": "c",
        "psi": "ψ",
        "Rf": "R<sub>f</sub>",
        "K0": "K<sub>0</sub>",
        "G0": "G<sub>0</sub>",
        "m": "m",
        "gamma07": "γ<sub>0.7</sub>",
        "nu": "ν",
        "pref": "p<sub>ref</sub>",
    }
    
    # 两列布局
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.subheader("⚙️ 参数设置")
        soil_type = st.selectbox("土体类型", ["黏土", "砂土"], index=0)
        
        st.markdown("#### 输入参数设置")
        st.markdown("（孔隙比 e 为必选，其余参数可自由启用/禁用）")
        
        input_dict = {}
        
        # e 强制启用
        st.checkbox("启用", value=True, disabled=True, key="enable_e")
        e_val = st.number_input("孔隙比 e", value=0.7, format="%.3f", key="hss_e")
        input_dict["e"] = e_val
        
        # Es - 使用HTML标签显示下标
        enable_Es = st.checkbox("启用", value=True, key="enable_Es")
        st.markdown("压缩模量 E<sub>s</sub> (MPa)", unsafe_allow_html=True)
        Es_val = st.number_input(
            "", value=10.0, format="%.2f", key="hss_Es", disabled=not enable_Es, label_visibility="collapsed"
        )
        if enable_Es:
            input_dict["Es"] = Es_val
        
        # sigma
        enable_sigma = st.checkbox("启用", value=True, key="enable_sigma")
        sigma_val = st.number_input(
            "竖向应力 σ (kPa)", value=100.0, format="%.2f", key="hss_sigma",
            disabled=not enable_sigma
        )
        if enable_sigma:
            input_dict["sigma"] = sigma_val
        
        # ps - 使用HTML标签显示下标
        enable_ps = st.checkbox("启用", value=True, key="enable_ps")
        st.markdown("比贯入阻力 p<sub>s</sub> (MPa)", unsafe_allow_html=True)
        ps_val = st.number_input(
            "", value=10.0, format="%.1f", key="hss_ps", disabled=not enable_ps, label_visibility="collapsed"
        )
        if enable_ps:
            input_dict["ps"] = ps_val
        
        calc_btn = st.button("🚀 计算 HSS 参数", type="primary", use_container_width=True)
    
    with col_right:
        if calc_btn:
            # 记录计算
            record_calculation_db()
            
            with st.spinner("正在加载..."):
                try:
                    equations, outputs = hss.predict(soil_type, input_dict)
                except Exception as e:
                    st.error(f"HSS 预测失败: {e}")
                    equations, outputs = {}, {}
            
            if outputs:
                # 对粘聚力 c 进行非负处理
                if 'c' in outputs and outputs['c'] < 0:
                    outputs['c'] = 0.0
                
                st.subheader("📐 HSS 模型参数")
                data = []
                for name, value in outputs.items():
                    unit = next((out.unit for out in hss.outputs if out.name == name), "")
                    meaning = param_meaning.get(name, "")
                    display_name = param_display_names.get(name, name)
                    
                    # 根据参数名称格式化数值
                    if name == "gamma07":
                        # 保持科学计数法，两位有效数字
                        val_str = f"{value:.2e}"
                    elif name == "phi":
                        # 保留一位小数
                        val_str = f"{value:.1f}"
                    elif name == "c":
                        # 保留一位小数
                        val_str = f"{value:.1f}"
                    elif name == "psi":
                        # 保留一位小数
                        val_str = f"{value:.1f}"
                    elif name == "G0":
                        # 保留一位小数
                        val_str = f"{value:.1f}"
                    elif name == "nu":
                        # 保留一位小数
                        val_str = f"{value:.1f}"
                    elif name == "pref":
                        # 保留一位小数
                        val_str = f"{value:.1f}"
                    elif name == "m":
                        # 保留两位小数
                        val_str = f"{value:.2f}"
                    elif name in ["Eoed", "E50", "Eur", "Rf", "K0"]:
                        # 保留两位小数
                        val_str = f"{value:.2f}"
                    else:
                        # 默认保留两位小数
                        val_str = f"{value:.2f}"
                    
                    data.append({
                        "参数": display_name,
                        "含义": meaning,
                        "预测值": val_str,
                        "单位": unit,
                    })
                df_hss = pd.DataFrame(data)
                # 使用HTML表格显示，以支持HTML下标标签
                st.markdown(df_to_html_table(df_hss, escape_html=False), unsafe_allow_html=True)
            else:
                st.warning("未能获取 HSS 预测结果，请检查输入参数。")
        else:
            st.info("👈 请在左侧设置参数后点击「计算」")
