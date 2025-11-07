import pandas as pd
import numpy as np

class EnhancedFeatureEngineer:
    def __init__(self):
        pass

    def _ensure_datetime(self, df, col='日期'):
        if col in df.columns:
            df = df.copy()
            df[col] = pd.to_datetime(df[col])
            return df
        else:
            raise ValueError(f"列 '{col}' 不存在于 DataFrame 中。")

    def create_temporal_features(self, dates_df):
        df = dates_df.copy().drop_duplicates().reset_index(drop=True)
        df['日期'] = pd.to_datetime(df['日期'])
        df['day_of_week'] = df['日期'].dt.dayofweek
        df['day_of_month'] = df['日期'].dt.day
        df['month'] = df['日期'].dt.month
        df['quarter'] = df['日期'].dt.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)
        return df

    def create_equipment_features(self, equipment_df, status_df):
        df = status_df.merge(
            equipment_df[['设备ID', '理论速度(瓶/天)', '正常速度(瓶/天)', '效率', '区域ID', '工序类型']],
            on='设备ID', how='left'
        )
        df['availability_ratio'] = df['运行时长(小时)'] / 16.0
        df['output_per_hour'] = df['当日产量(瓶)'] / df['运行时长(小时)'].replace(0, np.nan)
        df['capacity_utilization'] = df['当日产量(瓶)'] / df['理论速度(瓶/天)']
        df['efficiency_gap'] = df['效率'] - (df['当日产量(瓶)'] / df['正常速度(瓶/天)'])
        df['is_downtime'] = (df['停机时长(小时)'] > 0).astype(int)
        df = df.sort_values(['设备ID', '日期']).reset_index(drop=True)

        lag_cols = ['capacity_utilization', 'efficiency_gap', 'output_per_hour']
        for col in lag_cols:
            for lag in [1, 3]:
                df[f'{col}_lag{lag}'] = df.groupby('设备ID')[col].shift(lag)
        for col in lag_cols:
            df[f'{col}_roll3_mean'] = df.groupby('设备ID')[col].transform(lambda x: x.rolling(3, min_periods=1).mean())
            df[f'{col}_roll3_std'] = df.groupby('设备ID')[col].transform(lambda x: x.rolling(3, min_periods=1).std())
        df['efficiency_trend_3d'] = df.groupby('设备ID')['efficiency_gap'].transform(
            lambda x: x.diff().rolling(3, min_periods=1).mean()
        )
        return df

    def create_buffer_features(self, buffer_df):
        df = buffer_df.copy()
        df['buffer_turnover'] = df['出库数量(盘)'] / (df['期初数量(盘)'] + 1)
        df['inventory_coverage'] = df['期初数量(盘)'] / (df['出库数量(盘)'].replace(0, 1))
        df['safety_stock_ratio'] = df['期初数量(盘)'] / (df['安全库存(盘)'] + 1e-6)
        df['near_overflow'] = (df['safety_stock_ratio'] > 0.9).astype(int)
        df['near_empty'] = (df['safety_stock_ratio'] < 0.1).astype(int)

        df = df.sort_values(['缓冲区ID', '日期'])
        df['buffer_change'] = df.groupby('缓冲区ID')['期初数量(盘)'].diff()
        df['buffer_depletion_rate'] = df.groupby('缓冲区ID')['buffer_change'].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        return df

    def create_process_balance_features(self, production_df):
        proc_pivot = production_df.pivot_table(
            index='日期',
            columns='工序类型',
            values='总产量(瓶)',
            aggfunc='sum'
        ).reset_index()
        for col in ['灌装', '灭菌', '包装']:
            if col not in proc_pivot.columns:
                proc_pivot[col] = 0
        proc_pivot['fill_to_ster_ratio'] = proc_pivot['灭菌'] / (proc_pivot['灌装'] + 1e-6)
        proc_pivot['ster_to_pack_ratio'] = proc_pivot['包装'] / (proc_pivot['灭菌'] + 1e-6)
        proc_pivot['fill_to_pack_ratio'] = proc_pivot['包装'] / (proc_pivot['灌装'] + 1e-6)
        proc_pivot['bottleneck_score_fill'] = proc_pivot['灌装'] / (proc_pivot[['灭菌', '包装']].min(axis=1) + 1e-6)
        proc_pivot['bottleneck_score_ster'] = proc_pivot['灭菌'] / (proc_pivot['包装'] + 1e-6)
        proc_pivot['is_bottleneck_fill'] = (proc_pivot['bottleneck_score_fill'] > 1.1).astype(int)
        proc_pivot['is_bottleneck_ster'] = (proc_pivot['bottleneck_score_ster'] > 1.1).astype(int)
        return proc_pivot

    def create_labels(self, status_df, buffer_feat, process_balance_df):
        labels = pd.DataFrame({'日期': pd.to_datetime(status_df['日期'].unique())})

        # 停机预警
        downtime_next = status_df.copy()
        downtime_next['next_downtime'] = downtime_next.groupby('设备ID')['停机时长(小时)'].shift(-1) > 0
        label_downtime = downtime_next.groupby('日期')['next_downtime'].any().reset_index()
        labels = labels.merge(label_downtime, on='日期', how='left')

        # 缓冲区短缺预警
        buffer_next = buffer_feat.copy()
        buffer_next['next_near_empty'] = buffer_next.groupby('缓冲区ID')['near_empty'].shift(-1)
        label_buffer = buffer_next.groupby('日期')['next_near_empty'].any().reset_index()
        labels = labels.merge(label_buffer, on='日期', how='left')

        # 瓶颈标签
        labels = labels.merge(
            process_balance_df[['日期', 'is_bottleneck_fill', 'is_bottleneck_ster']],
            on='日期', how='left'
        )

        label_cols = ['next_downtime', 'next_near_empty', 'is_bottleneck_fill', 'is_bottleneck_ster']
        for col in label_cols:
            if col in labels.columns:
                labels[col] = labels[col].fillna(0).astype(int)
        return labels

    def aggregate_to_daily(self, equipment_features, buffer_features, process_features, labels):
        all_dates = pd.concat([
            equipment_features[['日期']],
            buffer_features[['日期']],
            process_features[['日期']],
            labels[['日期']]
        ]).drop_duplicates().reset_index(drop=True)
        feature_matrix = self.create_temporal_features(all_dates)

        # 设备聚合
        equip_agg = equipment_features.groupby('日期').agg({
            'capacity_utilization': ['mean', 'std', 'max', 'min'],
            'efficiency_gap': ['mean', 'std', 'max'],
            'availability_ratio': ['mean', 'min'],
            'is_downtime': 'sum',
            'output_per_hour': 'mean',
            'efficiency_trend_3d': 'mean'
        }).round(6)
        equip_agg.columns = ['equip_' + '_'.join(col).strip() for col in equip_agg.columns]
        equip_agg = equip_agg.reset_index()

        # 区域聚合
        area_agg = equipment_features.groupby(['日期', '区域ID']).agg({
            'capacity_utilization': 'mean'
        }).unstack('区域ID')
        area_agg.columns = [f'area_{col[1]}_util' for col in area_agg.columns]
        area_agg = area_agg.reset_index()

        feature_matrix = feature_matrix.merge(equip_agg, on='日期', how='left')
        feature_matrix = feature_matrix.merge(area_agg, on='日期', how='left')

        # 缓冲区聚合（按日期平均）
        buffer_daily = buffer_features.groupby('日期').agg({
            'buffer_turnover': 'mean',
            'inventory_coverage': 'mean',
            'safety_stock_ratio': 'mean',
            'near_overflow': 'max',
            'near_empty': 'max',
            'buffer_depletion_rate': 'mean'
        }).reset_index()
        feature_matrix = feature_matrix.merge(buffer_daily, on='日期', how='left')

        # 工序平衡特征（直接合并）
        feature_matrix = feature_matrix.merge(
            process_features[['日期', 'fill_to_pack_ratio', 'bottleneck_score_fill', 'bottleneck_score_ster']],
            on='日期', how='left'
        )

        # 合并标签
        feature_matrix = feature_matrix.merge(labels, on='日期', how='left')

        # 🔥 新增：回归目标（LSTM 所需）
        feature_matrix['production_gap'] = feature_matrix['equip_capacity_utilization_mean'] - 0.85
        feature_matrix['filling_packaging_balance'] = feature_matrix['fill_to_pack_ratio']
        feature_matrix['bottleneck_severity'] = feature_matrix[['bottleneck_score_fill', 'bottleneck_score_ster']].max(axis=1)
        feature_matrix['buffer_risk_score'] = 1.0 - feature_matrix['safety_stock_ratio']

        # 排序 & 填充缺失
        feature_matrix = feature_matrix.sort_values('日期').reset_index(drop=True)
        feature_matrix = feature_matrix.fillna(method='ffill').fillna(0)

        return feature_matrix

    def run_pipeline(self,
                    equipment_file='equipment_base.csv',
                    status_file='equipment_status_daily.csv',
                    buffer_file='buffer_inventory_daily.csv',
                    production_file='production_daily.csv'):
        print("🔍 正在加载数据...")
        equip_df = pd.read_csv(equipment_file)
        status_df = self._ensure_datetime(pd.read_csv(status_file))
        buffer_df = self._ensure_datetime(pd.read_csv(buffer_file))
        prod_df = self._ensure_datetime(pd.read_csv(production_file))

        print("⚙️ 正在构建设备特征...")
        equip_feat = self.create_equipment_features(equip_df, status_df)

        print("📦 正在构建缓冲区特征...")
        buffer_feat = self.create_buffer_features(buffer_df)

        print("⚖️ 正在构建工序平衡特征...")
        process_feat = self.create_process_balance_features(prod_df)

        print("🎯 正在生成预测标签...")
        labels = self.create_labels(status_df, buffer_feat, process_feat)

        print("🧩 正在聚合特征矩阵...")
        final_features = self.aggregate_to_daily(equip_feat, buffer_feat, process_feat, labels)

        print(f"✅ 特征工程完成！特征维度: {final_features.shape}")
        return final_features


if __name__ == "__main__":
    engineer = EnhancedFeatureEngineer()
    features = engineer.run_pipeline()
    features.to_csv('enhanced_production_features.csv', index=False, encoding='utf-8-sig')
    print("\n💾 特征矩阵已保存至: enhanced_production_features.csv")