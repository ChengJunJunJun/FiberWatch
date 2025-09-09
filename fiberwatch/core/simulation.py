
import numpy as np

def make_synthetic_otdr(n=4000, L_km=20.0, alpha_db_per_km=0.25, seed=1):
    """
    生成一条合成 OTDR 曲线（dB，越小越低），包含多个典型事件：
    - splice（非反射台阶）
    - connector（反射 + 台阶）
    - bend（斜率上升）
    - break（强反射 + 尾段噪声地板）
    """
    rng = np.random.default_rng(seed)
    z = np.linspace(0, L_km, n)
    baseline = -70 - alpha_db_per_km * z  # 简化的线性衰减
    y = baseline.copy()

    def add_splice(z_km, loss_db=0.25):
        i = np.searchsorted(z, z_km)
        y[i:] -= loss_db

    def add_connector(z_km, refl_db=3.0, loss_db=0.1):
        i = np.searchsorted(z, z_km)
        # 反射尖峰（向上数dB，即数值更高）
        y[max(0,i-2):min(n,i+3)] += refl_db
        y[i:] -= loss_db

    def add_bend(z_km, delta_alpha=0.15):
        i = np.searchsorted(z, z_km)
        for k in range(i, n):
            y[k] -= delta_alpha * (z[k]-z[i])

    def add_break(z_km, refl_db=8.0):
        i = np.searchsorted(z, z_km)
        y[max(0,i-2):min(n,i+3)] += refl_db
        # 之后进入噪声地板
        y[i+5:] = -95 + rng.normal(0, 1.0, size=n-(i+5))

    # 放置事件
    add_splice(3.2, 0.3)
    add_connector(7.8, refl_db=2.5, loss_db=0.15)
    add_connector(10.2, refl_db=1.5, loss_db=0.1)
    add_connector(12.4, refl_db=8.0, loss_db=0.5)  # 脏连接器
    add_bend(15.6, delta_alpha=0.08)
    add_break(18.9, refl_db=12.0)

    # 噪声
    y += np.interp(np.linspace(0,1,n), [0,1], [0.05, 0.3]) * rng.normal(0,1.0,n)

    return z, baseline, y
