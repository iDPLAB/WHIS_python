import numpy as np
from dataclasses import dataclass, field
from typing import List
import numpy as np


class ACFstatus:
    def __init__(self):
        self.num_ch = []          # 通道数
        self.num_filt = []        # 滤波器数量
        self.lbz = []             # 零点滤波器状态
        self.lap = []             # 极点滤波器状态
        self.sig_in_prev = []     # 上一时刻输入信号
        self.sig_out_prev = []    # 上一时刻输出信号
        self.count = []           # 处理计数器


class ACFcoef:
    def __init__(self):
        self.fs = []              # 采样频率
        self.ap = np.array([])    # 极点系数
        self.bz = np.array([])    # 零点系数


class cGCresp:
    def __init__(self):
        self.fr1 = []             # 第一共振频率
        self.n = []               # 滤波器阶数
        self.b1 = []              # 滤波器参数 b1
        self.c1 = []              # 滤波器参数 c1
        self.frat = []            # 频率比
        self.b2 = []              # 滤波器参数 b2
        self.c2 = []              # 滤波器参数 c2
        self.n_frq_rsl = []       # 频率分辨率点数
        self.pgc_frsp = []        # 被动 GC 频率响应
        self.cgc_frsp = []        # 压缩 GC 频率响应
        self.cgc_nrm_frsp = []    # 归一化压缩 GC 频率响应
        self.acf_frsp = []        # ACF 频率响应
        self.asym_func = []       # 非对称函数
        self.fp1 = []             # 第一峰值频率
        self.fr2 = []             # 第二共振频率
        self.fp2 = []             # 第二峰值频率
        self.val_fp2 = []         # 第二峰值幅值
        self.norm_fct_fp2 = []    # 第二峰值归一化系数
        self.freq = []            # 频率轴


class GCresp:
    def __init__(self):
        self.fr1 = []             # 第一共振频率
        self.fr2 = []             # 第二共振频率
        self.erb_space1 = []      # ERB 频率间隔
        self.ef = []              # 等效频率
        self.b1_val = []          # b1 参数值
        self.c1_val = []          # c1 参数值
        self.fp1 = []             # 第一峰值频率
        self.fp2 = []             # 第二峰值频率
        self.b2_val = []          # b2 参数值
        self.c1_val = []          # c1 参数值（重复定义）
        self.c2_val = []          # c2 参数值
        self.frat_val = []        # 频率比值
        self.frat0_val = []       # 初始频率比
        self.frat1_val = []       # 调整后的频率比
        self.pc_hpaf = []         # HPAF 压缩参数
        self.frat0_pc = []        # HPAF 前的频率比


class LvlEst:
    def __init__(self):
        self.lct_erb = []         # ERB 位置
        self.decay_hl = []        # 听力损失衰减
        self.b2 = []              # b2 参数
        self.c2 = []              # c2 参数
        self.frat = []            # 频率比
        self.rms2spldb = []       # RMS 到 SPL 的转换（dB）
        self.weight = []          # 权重
        self.ref_db = []          # 参考电平（dB）
        self.pwr = []             # 功率估计
        self.exp_decay_val = []   # 指数衰减值
        self.erb_space1 = []      # ERB 频率间隔
        self.n_ch_shift = []      # 通道移位数
        self.n_ch_lvl_est = []    # 电平估计通道数
        self.lvl_lin_min_lim = [] # 线性电平最小限制
        self.lvl_lin_ref = []     # 线性电平参考值


@dataclass
class DynHPAF:
    str_prc: str = 'frame-base'   # 处理方式（帧处理 / 逐样本），gcfb_v234 默认
    t_frame: float = 0.001        # 帧长度（秒），1 ms
    t_shift: float = 0.0005       # 帧移（秒），0.5 ms
    len_frame: List[float] = field(default_factory=list)  # 帧长度（样本点）
    len_shift: List[float] = field(default_factory=list)  # 帧移（样本点）
    fs: List[float] = field(default_factory=list)         # 采样频率
    name_win: str = 'hanning'     # 窗函数名称
    val_win: List[float] = field(default_factory=list)    # 窗函数数值


@dataclass
class HLoss:
    f_audgram_list: List[float] = field(
        default_factory=lambda: np.array([125, 250, 500, 1000, 2000, 4000, 8000])
    )                             # 听力图对应的频率列表（Hz）
    type: str = 'NH_NormalHearing'  # 听力损失类型（默认正常听力）
    hearing_level_db: List[float] = field(default_factory=list)          # 听力级（dB HL）
    pin_loss_db_act: List[float] = field(default_factory=list)           # 主动耳蜗损失（dB）
    pin_loss_db_act_init: List[float] = field(default_factory=list)      # 初始主动耳蜗损失（dB）
    pin_loss_db_pas: List[float] = field(default_factory=list)           # 被动耳蜗损失（dB）
    io_func_loss_db_pas: List[float] = field(default_factory=list)       # 被动损失的输入输出函数
    compression_health: List[float] = field(default_factory=list)        # 压缩健康度
    af_gain_cmpnst_db: List[float] = field(default_factory=list)         # 听觉滤波增益补偿
    hl_val_pin_cochlea_db: List[float] = field(default_factory=list)     # 耳蜗听力损失
    fb_fr1: List[float] = field(default_factory=list)                    # 滤波器组中心频率
    fb_hearing_level_db: List[float] = field(default_factory=list)       # 滤波器组听力级
    fb_pin_cochlea_db: List[float] = field(default_factory=list)         # 滤波器组耳蜗损失
    fb_pin_loss_db_act: List[float] = field(default_factory=list)        # 滤波器组主动损失
    fb_pin_loss_db_act_init: List[float] = field(default_factory=list)   # 滤波器组初始主动损失
    fb_pin_loss_db_pas: List[float] = field(default_factory=list)        # 滤波器组被动损失
    fb_compression_health: List[float] = field(default_factory=list)     # 滤波器组压缩健康度
    compression_health_initval: List[float] = field(default_factory=list)# 初始压缩健康度
    fb_af_gain_cmpnst_db: List[float] = field(default_factory=list)      # 滤波器组增益补偿


@dataclass
class CalibTone:
    SPLdB: int = None              # 校准音声压级（dB SPL）
    Tsnd: int = None               # 声音时长
    Freq: int = None               # 频率
    RMSDigitalLeveldB: int = None  # 数字 RMS 电平（dB）
    Ttaper: float = None           # 渐入渐出时间
    Name: str = None               # 校准音名称


@dataclass
class SrcSnd:
    SPLdB: int = None                               # 声源声压级
    RMSDigitalLevelStrWeight: str = None            # RMS 权重字符串
    RMSDigitalLeveldB: float = None                 # RMS 数字电平（dB）
    StrNormalizeWeight: str = None                  # 归一化权重方式
    SndLoad_RMSDigitalLeveldB: float = None         # 加载声音时的 RMS 电平
    RecordedCalibTone_RMSDigitalLeveldB: float = None # 录音校准音 RMS 电平


@dataclass
class GCparam:
    fs: int = 48000                 # 采样频率
    num_ch: int = 100               # 通道数
    f_range: np.ndarray = None      # 中心频率范围
    out_mid_crct: str = None        # 外耳 / 中耳校正方式
    ctrl: str = "dynamic"           # 动态或静态控制
    dyn_hpaf_str_prc: str = 'frame-base'  # HPAF 处理方式
    hloss_type: str = 'NH'          # 听力损失类型
    hloss: HLoss = field(default_factory=HLoss)
    TdelayFilt1kHz: float = None    # 1kHz 滤波器延迟
    TdelayFB: float = None          # 滤波器组延迟
    NumCmpnst = []                  # 补偿数量


@dataclass
class WHISparam:
    fs: int = 48000                 # 采样频率
    num_ch: int = 100               # 通道数
    f_range: np.ndarray = field(default_factory=lambda: np.array([100, 6000]))  # 频率范围
    out_mid_crct: str = "No"        # 是否进行外耳 / 中耳校正
    ctrl: str = "dynamic"           # 动态控制
    dyn_hpaf_str_prc: str = 'frame-base'  # HPAF 处理方式
    hloss_type: str = 'NH'          # 听力损失类型
    hloss: HLoss = field(default_factory=HLoss)
    gcparam: GCparam = field(default_factory=GCparam)     # 正常 GC 参数
    gcparamhl: GCparam = field(default_factory=GCparam)   # 听损 GC 参数
    calibtone: CalibTone = field(default_factory=CalibTone) # 校准音参数
    srcsnd: SrcSnd = field(default_factory=SrcSnd)         # 输入声音参数
    synth_method: str = None        # 合成方法
    sw_gui_batch: str = None        # GUI / batch 模式
    swplot: int = None              # 绘图开关
    allow_down_sampling: int = None # 是否允许降采样
    rate_down_sampling: int = None  # 降采样倍率
    fsorig: int = None              # 原始采样率
    Eqlz2MeddisHCLevel_AmpdB = []    # Meddis 毛细胞模型电平等化
    version: str = None             # 版本号
    TVFparam = {}                   # 时变滤波参数
    EMLoss = None                   # 中耳损失
