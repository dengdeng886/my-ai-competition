# -*- coding: utf-8 -*-
"""
基于 enhanced_production_features.csv 训练 LSTM 和 XGBoost 多目标回归模型
预测目标：
  - production_gap
  - filling_packaging_balance
  - bottleneck_severity
  - buffer_risk_score
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

# ==============================
# 1. 加载特征工程结果
# ==============================
FEATURE_FILE = 'enhanced_production_features.csv'
if not os.path.exists(FEATURE_FILE):
    raise FileNotFoundError(f"请先运行特征工程生成 {FEATURE_FILE}")

df = pd.read_csv(FEATURE_FILE, parse_dates=['日期'])
df = df.sort_values('日期').reset_index(drop=True)
print(f"✅ 加载数据完成，共 {len(df)} 行，{df.shape[1]} 列")

# ==============================
# 2. 定义特征与目标
# ==============================
TARGET_COLS = [
    'production_gap',
    'filling_packaging_balance',
    'bottleneck_severity',
    'buffer_risk_score'
]

# 排除日期和标签列（包括分类标签和目标列）
EXCLUDE_COLS = ['日期'] + [
    'next_downtime', 'next_near_empty', 'is_bottleneck_fill', 'is_bottleneck_ster'
] + TARGET_COLS

FEATURE_COLS = [col for col in df.columns if col not in EXCLUDE_COLS]
print(f"🎯 特征数量: {len(FEATURE_COLS)}")
print(f"🎯 回归目标: {TARGET_COLS}")

X = df[FEATURE_COLS].values.astype(np.float32)
y = df[TARGET_COLS].values.astype(np.float32)

# ==============================
# 3. 数据划分（按时间顺序）
# ==============================
# 保留最后 30 天作为测试集（时间序列不能随机打乱）
TEST_DAYS = 30
split_idx = len(df) - TEST_DAYS

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"📊 训练集: {X_train.shape}, 测试集: {X_test.shape}")

# ==============================
# 4. XGBoost 多输出回归
# ==============================
print("\n🚀 训练 XGBoost 多输出回归模型...")

xgb_model = MultiOutputRegressor(
    xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror'
    ),
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

# 评估
y_pred_xgb = xgb_model.predict(X_test)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"✅ XGBoost - MAE: {mae_xgb:.4f}, R²: {r2_xgb:.4f}")

# 保存
joblib.dump(xgb_model, 'xgb_multioutput_model.pkl')
print("💾 XGBoost 模型已保存为 xgb_multioutput_model.pkl")

# ==============================
# 5. LSTM 多输出回归
# ==============================
print("\n🧠 训练 LSTM 多输出回归模型...")

# 标准化（LSTM 对尺度敏感）
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# 构建时间序列滑窗：用过去 SEQ_LEN 天预测第 SEQ_LEN+1 天
SEQ_LEN = 7  # 使用过去7天预测未来1天

def create_sequences(X, y, seq_len):
    xs, ys = [], []
    for i in range(seq_len, len(X)):
        xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, SEQ_LEN)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, SEQ_LEN)

print(f"🧩 LSTM 输入形状: {X_train_seq.shape} → {y_train_seq.shape}")

# 构建模型
input_layer = Input(shape=(SEQ_LEN, X_train_seq.shape[2]))
x = LSTM(64, return_sequences=True)(input_layer)
x = Dropout(0.3)(x)
x = LSTM(32)(x)
x = Dropout(0.2)(x)
output_layer = Dense(len(TARGET_COLS), activation='linear')(x)

lstm_model = Model(inputs=input_layer, outputs=output_layer)
lstm_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# 回调
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 训练
history = lstm_model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_test_seq, y_test_seq),
    epochs=100,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# 预测（反标准化）
y_pred_scaled = lstm_model.predict(X_test_seq)
y_pred_lstm = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test_seq)

mae_lstm = mean_absolute_error(y_test_actual, y_pred_lstm)
r2_lstm = r2_score(y_test_actual, y_pred_lstm)

print(f"✅ LSTM - MAE: {mae_lstm:.4f}, R²: {r2_lstm:.4f}")

# 保存
lstm_model.save('lstm_multioutput_model.keras')
joblib.dump(scaler_X, 'lstm_scaler_X.pkl')
joblib.dump(scaler_y, 'lstm_scaler_y.pkl')
print("💾 LSTM 模型和标准化器已保存")

# ==============================
# 6. 结果对比
# ==============================
print("\n" + "="*50)
print("📈 模型性能对比（测试集）")
print("="*50)
print(f"{'模型':<12} | {'MAE':<10} | {'R²':<10}")
print("-"*50)
print(f"{'XGBoost':<12} | {mae_xgb:<10.4f} | {r2_xgb:<10.4f}")
print(f"{'LSTM':<12} | {mae_lstm:<10.4f} | {r2_lstm:<10.4f}")
print("="*50)

# 可选：保存预测结果示例
result_df = pd.DataFrame({
    '日期': df['日期'].iloc[-len(y_test_actual):].values,
    '真实_production_gap': y_test_actual[:, 0],
    'LSTM_pred_gap': y_pred_lstm[:, 0],
    'XGB_pred_gap': y_pred_xgb[-len(y_test_actual):, 0]
})
result_df.to_csv('prediction_comparison.csv', index=False, encoding='utf-8-sig')
print("\n📊 预测对比已保存至 prediction_comparison.csv")