# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import os
from sklearn.linear_model import LinearRegression

# ✅ 新增 Prophet 导入
from prophet import Prophet

warnings.filterwarnings("ignore")


# ==============================
# 1. 加载真实数据（使用脚本所在目录作为基准路径）
# ==============================
@st.cache_data
def load_real_data():
    """加载所有真实数据集，使用脚本所在目录作为基准路径"""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        equipment_df = pd.read_csv(os.path.join(base_dir, "equipment_base.csv"), encoding='utf-8')
        status_df = pd.read_csv(os.path.join(base_dir, "equipment_status_daily.csv"), parse_dates=['日期'], encoding='utf-8')
        efficiency_df = pd.read_csv(os.path.join(base_dir, "equipment_efficiency_daily.csv"), parse_dates=['日期'], encoding='utf-8')
        buffer_df = pd.read_csv(os.path.join(base_dir, "buffer_inventory_daily.csv"), parse_dates=['日期'], encoding='utf-8')
        prod_df = pd.read_csv(os.path.join(base_dir, "production_daily.csv"), parse_dates=['日期'], encoding='utf-8')

        # 按日期聚合产量
        prod_agg = prod_df.groupby('日期').agg(
            计划产量_瓶=('计划产量(瓶)', 'first'),
            总产量_瓶=('总产量(瓶)', 'max')
        ).reset_index()
        prod_agg.rename(columns={'计划产量_瓶': '计划产量(瓶)', '总产量_瓶': '总产量(瓶)'}, inplace=True)
        prod_df = prod_agg

        return equipment_df, status_df, efficiency_df, buffer_df, prod_df

    except Exception as e:
        st.error(f"❌ 数据加载失败: {e}")
        st.stop()


# ==============================
# 2. 优化引擎
# ==============================
class OptimizationEngine:
    def __init__(self, equipment_df):
        self.equipment_df = equipment_df
        self.filler_ids = equipment_df[equipment_df['工序类型'] == '灌装']['设备ID'].tolist()
        self.packer_ids = equipment_df[equipment_df['工序类型'] == '包装']['设备ID'].tolist()

    def analyze_situation(self, current_features, predictions):
        risks = []
        recommendations = []

        buffer_score = predictions.get('buffer_risk_score', 0.5)
        if buffer_score > 0.8:
            risks.append("⚠️ 缓冲区库存过高，存在积压风险")
            recommendations.append("💡 建议提高下游包装线速度或暂停上游灌装")
        elif buffer_score < 0.2:
            risks.append("⚠️ 缓冲区库存不足，存在断料风险")
            recommendations.append("💡 建议加快上游灌装速度或启用备用设备")

        bottleneck = predictions.get('bottleneck_severity', 0.0)
        if bottleneck > 0.6:
            risks.append("⚠️ 发现严重产能瓶颈")
            recommendations.append("💡 建议启动备用包装线或调整排产计划")

        balance = predictions.get('filling_packaging_balance', 1.0)
        if abs(balance - 1.0) > 0.3:
            direction = "灌装过快" if balance > 1.0 else "包装过快"
            risks.append(f"⚠️ 灌装与包装产线失衡（{direction}）")
            recommendations.append("💡 建议动态调节灌装速度以匹配包装能力")

        gap = predictions.get('production_gap', 0.0)
        if gap > 0.2:
            risks.append("⚠️ 实际产量低于计划量，存在交付风险")
            recommendations.append("💡 建议加班或增加班次以弥补缺口")

        return {"risks": risks, "recommendations": recommendations}


# ==============================
# 3. 主程序类
# ==============================
class ProductionDashboard:
    def __init__(self, equipment_df, status_df, efficiency_df, buffer_df, prod_df):
        self.equipment_df = equipment_df
        self.status_df = status_df
        self.efficiency_df = efficiency_df
        self.buffer_df = buffer_df
        self.prod_df = prod_df
        self.optimization_engine = OptimizationEngine(equipment_df)

        # ✅ 关键修复：在这里定义 filler_ids 和 packer_ids
        self.filler_ids = equipment_df[equipment_df['工序类型'] == '灌装']['设备ID'].tolist()
        self.packer_ids = equipment_df[equipment_df['工序类型'] == '包装']['设备ID'].tolist()

        self.selected_date = None
        self.date_range = None
        self.SAFE_BUFFER = 2880  # 安全库存（盘）

    def _create_sidebar(self):
        st.sidebar.title("控制面板")
        self.date_range = st.sidebar.date_input(
            "分析时间范围",
            value=(datetime(2025, 6, 1), datetime(2025, 8, 12)),
            min_value=datetime(2025, 6, 1),
            max_value=datetime(2025, 8, 12)
        )
        if len(self.date_range) != 2:
            st.sidebar.warning("请选择一个日期范围")
            st.stop()
        self.selected_date = self.date_range[1]

        st.sidebar.slider("缓冲区低预警阈值", 0.1, 0.5, 0.2, 0.1, key="low_thresh")
        st.sidebar.slider("缓冲区高预警阈值", 0.5, 0.9, 0.8, 0.1, key="high_thresh")
        st.sidebar.selectbox("预测周期（天）", [3, 7, 14], index=0, key="pred_days")

    def _get_current_state(self):
        target_date = pd.Timestamp(self.selected_date)

        # 产量数据
        prod_row = self.prod_df[self.prod_df['日期'] == target_date]
        if prod_row.empty:
            st.error(f"⚠️ 日期 {self.selected_date} 在 production_daily.csv 中无数据")
            st.stop()
        prod_row = prod_row.iloc[0]
        plan_yield = prod_row['计划产量(瓶)']
        actual_yield = prod_row['总产量(瓶)']
        gap_ratio = max(0.0, (plan_yield - actual_yield) / plan_yield) if plan_yield > 0 else 0.0

        # 效率数据
        eff_row = self.efficiency_df[self.efficiency_df['日期'] == target_date]
        if eff_row.empty:
            oee = utilization = 0.0
        else:
            eff_row = eff_row.iloc[0]
            oee = eff_row['综合效率(OEE)']
            utilization = eff_row['产能利用率']

        # 灌装 vs 包装产量
        filler_output = self.status_df[
            (self.status_df['日期'] == target_date) &
            (self.status_df['设备ID'].isin(self.filler_ids))
        ]['当日产量(瓶)'].sum()

        packer_output = self.status_df[
            (self.status_df['日期'] == target_date) &
            (self.status_df['设备ID'].isin(self.packer_ids))
        ]['当日产量(瓶)'].sum()

        balance_ratio = filler_output / (packer_output + 1e-8)

        # 缓冲区库存
        buffer_row = self.buffer_df[self.buffer_df['日期'] == target_date]
        buffer_level = buffer_row['期末数量(盘)'].sum() if not buffer_row.empty else 0
        buffer_risk_score = min(buffer_level / self.SAFE_BUFFER, 1.2)  # 限制上限

        # 瓶颈严重度：基于平衡比偏离1的程度
        bottleneck_severity = abs(balance_ratio - 1.0) / (1.0 + abs(balance_ratio - 1.0))

        return {
            'safety_stock_ratio': 1.0,
            'buffer_risk_score': buffer_risk_score,
            'bottleneck_severity': bottleneck_severity,
            'filling_packaging_balance': balance_ratio,
            'production_gap': gap_ratio,
            'oee': oee,
            'utilization': utilization,
            'daily_output': int(actual_yield),
            'plan_yield': int(plan_yield),
            'buffer_inventory': int(buffer_level)
        }

    def _plot_production_trends(self):
        start, end = self.date_range
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        df = self.prod_df[(self.prod_df['日期'] >= start_ts) & (self.prod_df['日期'] <= end_ts)].copy()

        if df.empty:
            st.warning("所选日期范围内无生产数据")
            return

        fig = px.line(
            df, x='日期',
            y=['总产量(瓶)', '计划产量(瓶)'],
            labels={'value': '产量(瓶)', 'variable': '类型'},
            title="车间日产量趋势"
        )
        fig.add_hline(y=df['计划产量(瓶)'].mean(), line_dash="dash", line_color="red", annotation_text="平均计划")
        st.plotly_chart(fig, use_container_width=True)

    # ✅ 替换为 Prophet 预测
    def _plot_prophet_prediction(self):
        start, end = self.date_range
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        df = self.prod_df[(self.prod_df['日期'] >= start_ts) & (self.prod_df['日期'] <= end_ts)].copy()

        if df.empty:
            st.warning("所选日期范围内无生产数据，无法预测")
            return

        pred_days = st.session_state.get("pred_days", 3)

        # Prophet 要求列名为 ds 和 y
        prophet_df = df[['日期', '总产量(瓶)']].rename(columns={'日期': 'ds', '总产量(瓶)': 'y'})
        prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)

        # 创建并拟合模型
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.95
        )
        model.fit(prophet_df)

        # 构建未来日期
        future = model.make_future_dataframe(periods=pred_days, freq='D')
        forecast = model.predict(future)

        # 合并历史与预测
        hist_df = prophet_df.copy()
        hist_df['类型'] = '历史'

        pred_df = forecast[['ds', 'yhat']].tail(pred_days).copy()
        pred_df = pred_df.rename(columns={'ds': '日期', 'yhat': '总产量(瓶)'})
        pred_df['总产量(瓶)'] = np.maximum(pred_df['总产量(瓶)'], 0)  # 防止负值
        pred_df['类型'] = '预测'

        hist_df = hist_df.rename(columns={'ds': '日期', 'y': '总产量(瓶)'})

        plot_df = pd.concat([hist_df, pred_df], ignore_index=True)

        fig = px.line(
            plot_df,
            x='日期',
            y='总产量(瓶)',
            color='类型',
            title="日产量 Prophet 预测趋势",
            line_dash='类型',
            labels={'总产量(瓶)': '产量(瓶)'}
        )
        fig.update_traces(mode='lines+markers')
        st.plotly_chart(fig, use_container_width=True)

    def _show_buffer_analysis(self, state):
        start, end = self.date_range
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        df = self.buffer_df[(self.buffer_df['日期'] >= start_ts) & (self.buffer_df['日期'] <= end_ts)].copy()

        if df.empty:
            st.warning("无缓冲区数据")
            return

        df['库存占比'] = df['期末数量(盘)'] / self.SAFE_BUFFER
        fig = px.line(df, x='日期', y='库存占比', title="缓冲区库存占比趋势")
        fig.add_hline(y=st.session_state.get("high_thresh", 0.8), line_dash="dash", line_color="red", annotation_text="高预警")
        fig.add_hline(y=st.session_state.get("low_thresh", 0.2), line_dash="dash", line_color="green", annotation_text="低预警")
        st.plotly_chart(fig, use_container_width=True)

    def _show_alerts_warnings(self, risks):
        st.subheader("🚨 风险预警")
        if not risks:
            st.success("✅ 当前生产稳定，无重大风险")
        for risk in risks:
            st.markdown(f"<span style='color:red;'>{risk}</span>", unsafe_allow_html=True)

    def _show_optimization_recommendations(self, recommendations, state):
        st.subheader("💡 优化建议")
        if not recommendations:
            st.info("暂无优化建议")
        for rec in recommendations:
            st.markdown(f"<span style='color:#007BFF;'>{rec}</span>", unsafe_allow_html=True)

    def _show_production_overview(self, state):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📅 当前日期", self.selected_date.strftime('%Y-%m-%d'))
        with col2:
            st.metric("🏭 实际产量", f"{state['daily_output']:,} 瓶")
        with col3:
            st.metric("🎯 计划产量", f"{state['plan_yield']:,} 瓶")

        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("📊 产能利用率", f"{state['utilization'] * 100:.1f}%")
        with col5:
            st.metric("⚙️ 综合效率(OEE)", f"{state['oee'] * 100:.1f}%")
        with col6:
            st.metric("📦 缓冲区库存", f"{state['buffer_inventory']:,} 盘")

    def _show_detailed_analysis(self, report, state):
        st.subheader("🔍 详细分析")
        st.write("**当前状态总结**:")
        st.write(f"- 产能缺口: {state['production_gap'] * 100:.1f}%")
        st.write(f"- 灌装/包装平衡比: {state['filling_packaging_balance']:.2f}")
        st.write(f"- 缓冲区风险评分: {state['buffer_risk_score']:.2f} (安全库存={self.SAFE_BUFFER}盘)")

    def run_dashboard(self):
        st.set_page_config(
            page_title="智能生产调度系统",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.markdown("""
        <style>
            .stApp { overflow-y: auto; scrollbar-width: thin; -ms-overflow-style: scrollbar; }
            .stApp::-webkit-scrollbar { width: 8px; }
            .stApp::-webkit-scrollbar-track { background: #f1f1f1; }
            .stApp::-webkit-scrollbar-thumb { background: #c1c1c1; border-radius: 4px; }
            .stApp::-webkit-scrollbar-thumb:hover { background: #a8a8a8; }
            div[data-testid="stAppViewContainer"] { overflow-y: auto; }
            section[data-testid="stSidebar"] { overflow-y: auto; scrollbar-width: thin; }
            section[data-testid="stSidebar"]::-webkit-scrollbar { width: 6px; }
            section[data-testid="stSidebar"]::-webkit-scrollbar-thumb { background: #c1c1c1; border-radius: 3px; }
        </style>
        """, unsafe_allow_html=True)

        st.title("🏭 智能生产调度与产能优化系统")
        st.markdown("---")

        self._create_sidebar()
        current_state = self._get_current_state()

        predictions = {
            'production_gap': current_state['production_gap'],
            'filling_packaging_balance': current_state['filling_packaging_balance'],
            'buffer_risk_score': current_state['buffer_risk_score'],
            'bottleneck_severity': current_state['bottleneck_severity']
        }
        current_features = {'safety_stock_ratio': 1.0}

        report = self.optimization_engine.analyze_situation(current_features, predictions)

        # 主布局：左侧图表 + 右侧预警/建议
        col_main, col_side = st.columns([2, 1])
        with col_main:
            self._show_production_overview(current_state)
            self._plot_production_trends()
            # ✅ 使用 Prophet 预测替代线性回归
            self._plot_prophet_prediction()
        with col_side:
            self._show_alerts_warnings(report['risks'])
            self._show_optimization_recommendations(report['recommendations'], current_state)

        # 缓冲区分析单独一行，居中展示
        st.markdown("---")
        self._show_buffer_analysis(current_state)


# ==============================
# 4. 主函数入口
# ==============================
def main():
    equipment_df, status_df, efficiency_df, buffer_df, prod_df = load_real_data()
    dashboard = ProductionDashboard(equipment_df, status_df, efficiency_df, buffer_df, prod_df)
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()