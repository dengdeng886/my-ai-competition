# inference_module.py
import os
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==============================
# 1. 特征工程函数（精确匹配原始数据）
# ==============================
def build_features_from_raw(
    production_file: str,
    status_file: str,
    efficiency_file: str,
    base_file: str,
    buffer_file: str,
    target_date: str
) -> pd.DataFrame:
    """
    从5个原始CSV文件构建与 enhanced_production_features.csv 一致的特征
    :param target_date: 预测目标日期（字符串）
    :return: 单行 DataFrame，含30个特征 + 日期
    """
    target_date = pd.to_datetime(target_date)
    feat_date = target_date - timedelta(days=1)  # 使用前一天数据

    # --- 加载数据 ---
    prod_df = pd.read_csv(production_file, parse_dates=['日期'])
    status_df = pd.read_csv(status_file, parse_dates=['日期'])
    eff_df = pd.read_csv(efficiency_file, parse_dates=['日期'])
    base_df = pd.read_csv(base_file)
    buffer_df = pd.read_csv(buffer_file, parse_dates=['日期'])

    # --- 1. 时间特征 ---
    features = {
        '日期': target_date,
        'day_of_week': feat_date.dayofweek,
        'day_of_month': feat_date.day,
        'month': feat_date.month,
        'quarter': (feat_date.month - 1) // 3 + 1,
        'is_weekend': int(feat_date.dayofweek >= 5),
        'is_month_start': int(feat_date.day == 1),
        'is_month_end': int(feat_date.is_month_end),  # ✅ 修正：属性，无括号
    }

    # --- 2. 合并设备基础信息 ---
    status_df = status_df.merge(base_df, on='设备ID', how='left')
    eff_df = eff_df.merge(base_df, on='设备ID', how='left')

    # --- 3. 设备级特征计算 ---
    status_day = status_df[status_df['日期'] == feat_date].copy()
    eff_day = eff_df[eff_df['日期'] == feat_date].copy()

    if not eff_day.empty:
        # 产能利用率（实际 / 理论）
        eff_day['产能利用率'] = np.where(
            eff_day['理论产能(瓶)'] > 0,
            eff_day['实际产能(瓶)'] / eff_day['理论产能(瓶)'],
            0.0
        )
        # 运行效率（已存在）
        eff_day['运行效率'] = eff_day['运行效率'].fillna(0.0)
        # 效率 gap = 1 - 运行效率
        eff_day['效率_gap'] = 1.0 - eff_day['运行效率']
        # 可用率（来自基础表）
        eff_day['可用率'] = eff_day['可用率'].fillna(0.9)

        util_vals = eff_day['产能利用率'].values
        gap_vals = eff_day['效率_gap'].values
        avail_vals = eff_day['可用率'].values

        features.update({
            'equip_capacity_utilization_mean': float(np.mean(util_vals)),
            'equip_capacity_utilization_std': float(np.std(util_vals)) if len(util_vals) > 1 else 0.0,
            'equip_capacity_utilization_max': float(np.max(util_vals)),
            'equip_capacity_utilization_min': float(np.min(util_vals)),
            'equip_efficiency_gap_mean': float(np.mean(gap_vals)),
            'equip_efficiency_gap_std': float(np.std(gap_vals)) if len(gap_vals) > 1 else 0.0,
            'equip_efficiency_gap_max': float(np.max(gap_vals)),
            'equip_availability_ratio_mean': float(np.mean(avail_vals)),
            'equip_availability_ratio_min': float(np.min(avail_vals)),
            'equip_is_downtime_sum': int((eff_day['故障次数'] > 0).sum()),
            'equip_output_per_hour_mean': float(eff_day['实际产能(瓶)'].sum() / 24.0),
        })

        # 区域利用率
        area_util = eff_day.groupby('区域ID')['产能利用率'].mean()
        features['area_AREA_A_util'] = float(area_util.get('AREA_A', 0.0))
        features['area_AREA_B_util'] = float(area_util.get('AREA_B', 0.0))

        # 🔧 修正：灌装 vs 包装产量（安全计算 ratio）
        prod_day = prod_df[prod_df['日期'] == feat_date].copy()
        filler_prod = prod_day[prod_day['工序类型'] == '灌装']['总产量(瓶)'].sum()
        packer_prod = prod_day[prod_day['工序类型'] == '包装']['总产量(瓶)'].sum()

        # 安全计算 fill_to_pack_ratio
        if packer_prod <= 0:
            fill_to_pack_ratio = 1.0  # 默认平衡
        elif filler_prod <= 0:
            fill_to_pack_ratio = 0.0
        else:
            fill_to_pack_ratio = filler_prod / packer_prod
            # 限制合理范围，防止极端值影响模型
            fill_to_pack_ratio = max(0.0, min(fill_to_pack_ratio, 10.0))

        features['fill_to_pack_ratio'] = float(fill_to_pack_ratio)

        # 瓶颈分数（简化）
        filler_gap = eff_day[eff_day['工序类型'] == '灌装']['效率_gap'].mean()
        ster_gap = eff_day[eff_day['工序类型'] == '灭菌']['效率_gap'].mean()
        features['bottleneck_score_fill'] = float(filler_gap) if not pd.isna(filler_gap) else 0.0
        features['bottleneck_score_ster'] = float(ster_gap) if not pd.isna(ster_gap) else 0.0

    else:
        for k in ['equip_capacity_utilization_mean', 'equip_efficiency_gap_mean',
                  'area_AREA_A_util', 'area_AREA_B_util', 'fill_to_pack_ratio']:
            features[k] = 0.0
        features.update({
            'equip_capacity_utilization_std': 0.0,
            'equip_capacity_utilization_max': 0.0,
            'equip_capacity_utilization_min': 0.0,
            'equip_efficiency_gap_std': 0.0,
            'equip_efficiency_gap_max': 0.0,
            'equip_availability_ratio_mean': 0.9,
            'equip_availability_ratio_min': 0.9,
            'equip_is_downtime_sum': 0,
            'equip_output_per_hour_mean': 0.0,
            'bottleneck_score_fill': 0.0,
            'bottleneck_score_ster': 0.0,
        })

    # --- 4. 缓冲区特征 ---
    buffer_day = buffer_df[buffer_df['日期'] == feat_date].copy()
    if not buffer_day.empty:
        buffer_day['安全库存'] = buffer_day['安全库存(盘)']
        buffer_day['期末库存'] = buffer_day['期末数量(盘)']
        buffer_day['出库量'] = buffer_day['出库数量(盘)']
        buffer_day['期初库存'] = buffer_day['期初数量(盘)']

        total_out = buffer_day['出库量'].sum()
        total_begin = buffer_day['期初库存'].sum()
        total_end = buffer_day['期末库存'].sum()
        total_safety = buffer_day['安全库存'].sum()

        features['buffer_turnover'] = float(total_out / total_begin) if total_begin > 0 else 0.0
        features['inventory_coverage'] = float(total_end / (total_out / 24)) if total_out > 0 else 0.0
        features['safety_stock_ratio'] = float(total_end / total_safety) if total_safety > 0 else 1.0

        near_empty = (buffer_day['期末库存'] < 0.2 * buffer_day['安全库存']).any()
        near_overflow = (buffer_day['期末库存'] > 1.5 * buffer_day['安全库存']).any()
        features['near_empty'] = int(near_empty)
        features['near_overflow'] = int(near_overflow)
        features['buffer_depletion_rate'] = float((total_begin - total_end) / total_begin) if total_begin > 0 else 0.0
    else:
        for k in ['buffer_turnover', 'inventory_coverage', 'safety_stock_ratio',
                  'near_overflow', 'near_empty', 'buffer_depletion_rate']:
            features[k] = 0.0

    # --- 5. 效率趋势（过去3天）---
    try:
        dates_3d = [feat_date - timedelta(days=i) for i in range(3)]
        eff_3d = eff_df[eff_df['日期'].isin(dates_3d)].copy()
        if len(eff_3d) >= 1:
            eff_3d['产能利用率'] = np.where(
                eff_3d['理论产能(瓶)'] > 0,
                eff_3d['实际产能(瓶)'] / eff_3d['理论产能(瓶)'],
                0.0
            )
            trend = eff_3d.groupby('日期')['产能利用率'].mean().values
            if len(trend) >= 2:
                slope = np.polyfit(np.arange(len(trend)), trend, 1)[0]
                features['equip_efficiency_trend_3d_mean'] = float(slope)
            else:
                features['equip_efficiency_trend_3d_mean'] = 0.0
        else:
            features['equip_efficiency_trend_3d_mean'] = 0.0
    except Exception:
        features['equip_efficiency_trend_3d_mean'] = 0.0

    # --- 6. 补全所有30个特征 ---
    expected_features = [
        'day_of_week', 'day_of_month', 'month', 'quarter', 'is_weekend',
        'is_month_start', 'is_month_end',
        'equip_capacity_utilization_mean', 'equip_capacity_utilization_std',
        'equip_capacity_utilization_max', 'equip_capacity_utilization_min',
        'equip_efficiency_gap_mean', 'equip_efficiency_gap_std', 'equip_efficiency_gap_max',
        'equip_availability_ratio_mean', 'equip_availability_ratio_min',
        'equip_is_downtime_sum', 'equip_output_per_hour_mean',
        'equip_efficiency_trend_3d_mean',
        'area_AREA_A_util', 'area_AREA_B_util',
        'buffer_turnover', 'inventory_coverage', 'safety_stock_ratio',
        'near_overflow', 'near_empty', 'buffer_depletion_rate',
        'fill_to_pack_ratio', 'bottleneck_score_fill', 'bottleneck_score_ster'
    ]
    for f in expected_features:
        if f not in features:
            features[f] = 0.0

    return pd.DataFrame([features])


# ==============================
# 2. 推理主函数（支持 XGBoost + LSTM）
# ==============================
def run_inference(
    production_file: str = 'production_daily.csv',
    status_file: str = 'equipment_status_daily.csv',
    efficiency_file: str = 'equipment_efficiency_daily.csv',
    base_file: str = 'equipment_base.csv',
    buffer_file: str = 'buffer_inventory_daily.csv',
    target_date: str = None
) -> dict:
    """
    端到端推理：原始数据 → 特征 → XGBoost + LSTM → 融合预测
    """
    if target_date is None:
        target_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')

    # 构建特征
    features_df = build_features_from_raw(
        production_file, status_file, efficiency_file, base_file, buffer_file, target_date
    )

    FEATURE_COLS = [
        'day_of_week', 'day_of_month', 'month', 'quarter', 'is_weekend',
        'is_month_start', 'is_month_end',
        'equip_capacity_utilization_mean', 'equip_capacity_utilization_std',
        'equip_capacity_utilization_max', 'equip_capacity_utilization_min',
        'equip_efficiency_gap_mean', 'equip_efficiency_gap_std', 'equip_efficiency_gap_max',
        'equip_availability_ratio_mean', 'equip_availability_ratio_min',
        'equip_is_downtime_sum', 'equip_output_per_hour_mean',
        'equip_efficiency_trend_3d_mean',
        'area_AREA_A_util', 'area_AREA_B_util',
        'buffer_turnover', 'inventory_coverage', 'safety_stock_ratio',
        'near_overflow', 'near_empty', 'buffer_depletion_rate',
        'fill_to_pack_ratio', 'bottleneck_score_fill', 'bottleneck_score_ster'
    ]

    X = features_df[FEATURE_COLS].values.astype(np.float32)

    # --- 1. XGBoost 预测 ---
    xgb_model = joblib.load('xgb_multioutput_model.pkl')
    pred_xgb = xgb_model.predict(X)[0]

    # --- 2. LSTM 预测（需过去7天历史）---
    try:
        # 获取最近7天特征（用于LSTM）
        last_7_days = [pd.to_datetime(target_date) - timedelta(days=i) for i in range(7)]
        lstm_features = []
        for d in last_7_days:
            df = build_features_from_raw(
                production_file, status_file, efficiency_file, base_file, buffer_file, d.strftime('%Y-%m-%d')
            )
            lstm_features.append(df[FEATURE_COLS].values.flatten())
        X_lstm = np.array(lstm_features).reshape(1, 7, -1)

        # 加载 LSTM 模型和标准化器
        scaler_X = joblib.load('lstm_scaler_X.pkl')
        scaler_y = joblib.load('lstm_scaler_y.pkl')
        model = load_model('lstm_multioutput_model.keras')

        X_lstm_scaled = scaler_X.transform(X_lstm.reshape(-1, X_lstm.shape[-1])).reshape(X_lstm.shape)
        pred_lstm_scaled = model.predict(X_lstm_scaled, verbose=0)
        pred_lstm = scaler_y.inverse_transform(pred_lstm_scaled)[0]

        # --- 3. 融合策略：加权平均 ---
        final_pred = 0.7 * pred_xgb + 0.3 * pred_lstm  # XGBoost 主导，LSTM 提供趋势
    except Exception as e:
        print(f"LSTM 预测失败，使用 XGBoost: {e}")
        final_pred = pred_xgb

    return {
        'date': target_date,
        'predictions': {
            'production_gap': float(final_pred[0]),
            'filling_packaging_balance': float(final_pred[1]),
            'bottleneck_severity': float(final_pred[2]),
            'buffer_risk_score': float(final_pred[3])
        },
        'model_used': 'fusion' if 'final_pred' in locals() and np.array_equal(final_pred, 0.7*pred_xgb + 0.3*pred_lstm) else 'xgboost'
    }


# ==============================
# 3. 测试
# ==============================
if __name__ == "__main__":
    try:
        result = run_inference(target_date='2025-10-22')
        print("✅ 推理成功!")
        print(f"预测日期: {result['date']}")
        for k, v in result['predictions'].items():
            print(f"{k}: {v:.4f}")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()