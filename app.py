# -*- coding: utf-8 -*-
import streamlit as st


# 为避免命名冲突，将两个系统的 main 函数重命名
def run_oee_system():
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from scipy.stats import pearsonr
    import warnings
    import io

    warnings.filterwarnings('ignore')

    class OEEAnalyzer:
        def __init__(self, data):
            self.df = data.copy()
            self.df.columns = [col.strip() for col in self.df.columns]
            self.df['月份'] = pd.to_datetime(self.df['月份'])
            self.df['月份序号'] = range(1, len(self.df) + 1)
            self.df['时间'] = self.df['月份']
            st.success(f"数据加载成功，共{len(self.df)}条记录")
            st.info(f"数据时间范围: {self.df['时间'].min().strftime('%Y-%m')} 至 {self.df['时间'].max().strftime('%Y-%m')}")

        def quantitative_analysis(self):
            st.header("📊 定量分析结果 - 影响因素优先级排序")
            results = {}
            factors = ['设备有效利用率', '性能时间', '良品率']
            correlations = {}
            for factor in factors:
                corr, _ = pearsonr(self.df[factor], self.df['OEE'])
                correlations[factor] = abs(corr)
            corr_scores = pd.Series(correlations)
            if corr_scores.max() > corr_scores.min():
                corr_scores = (corr_scores - corr_scores.min()) / (corr_scores.max() - corr_scores.min())
            else:
                corr_scores = pd.Series({k: 0.5 for k in corr_scores.index})

            sensitivities = {}
            for factor in factors:
                try:
                    factor_changes = self.df[factor].pct_change().dropna()
                    oee_changes = self.df['OEE'].pct_change().dropna()
                    min_len = min(len(factor_changes), len(oee_changes))
                    factor_changes = factor_changes.iloc[:min_len]
                    oee_changes = oee_changes.iloc[:min_len]
                    valid_mask = (factor_changes != 0) & (~factor_changes.isna()) & (~oee_changes.isna())
                    if valid_mask.any():
                        sensitivity_values = (oee_changes[valid_mask] / factor_changes[valid_mask]).abs()
                        sensitivity = sensitivity_values.mean()
                        sensitivities[factor] = sensitivity if not np.isnan(sensitivity) else 0
                    else:
                        sensitivities[factor] = 0
                except:
                    sensitivities[factor] = 0

            sens_scores = pd.Series(sensitivities)
            if len(sens_scores) > 0 and sens_scores.max() > sens_scores.min():
                sens_scores = (sens_scores - sens_scores.min()) / (sens_scores.max() - sens_scores.min())
            else:
                sens_scores = pd.Series({k: 0.5 for k in factors})

            X = self.df[factors]
            y = self.df['OEE']
            model = LinearRegression()
            model.fit(X, y)
            reg_importance = pd.Series(
                np.abs(model.coef_) / np.sum(np.abs(model.coef_)),
                index=factors
            )

            contributions = {}
            for factor in factors:
                contrib_values = []
                for i, row in self.df.iterrows():
                    base_oee = row['设备有效利用率'] * row['性能时间'] * row['良品率']
                    if base_oee == 0:
                        contrib_values.append(0)
                        continue
                    temp_row = row.copy()
                    temp_row[factor] = temp_row[factor] * 1.01
                    new_oee = temp_row['设备有效利用率'] * temp_row['性能时间'] * temp_row['良品率']
                    contribution = (new_oee - base_oee) / base_oee * 100
                    contrib_values.append(abs(contribution))
                contributions[factor] = np.mean(contrib_values) if len(contrib_values) > 0 else 0

            contrib_scores = pd.Series(contributions)
            if contrib_scores.max() > contrib_scores.min():
                contrib_scores = (contrib_scores - contrib_scores.min()) / (contrib_scores.max() - contrib_scores.min())
            else:
                contrib_scores = pd.Series({k: 0.5 for k in factors})

            weights = {'correlation': 0.3, 'sensitivity': 0.3, 'regression': 0.2, 'contribution': 0.2}
            final_scores = {}
            for factor in factors:
                corr_score = corr_scores.get(factor, 0)
                sens_score = sens_scores.get(factor, 0)
                reg_score = reg_importance.get(factor, 0)
                contrib_score = contrib_scores.get(factor, 0)
                if np.isnan(corr_score): corr_score = 0
                if np.isnan(sens_score): sens_score = 0
                if np.isnan(reg_score): reg_score = 0
                if np.isnan(contrib_score): contrib_score = 0
                total_score = (
                        corr_score * weights['correlation'] +
                        sens_score * weights['sensitivity'] +
                        reg_score * weights['regression'] +
                        contrib_score * weights['contribution']
                )
                final_scores[factor] = total_score

            ranking = pd.Series(final_scores).sort_values(ascending=False)

            st.subheader("1.各分析方法详细得分")
            analysis_df = pd.DataFrame({
                '相关性得分': [corr_scores.get(f, 0) for f in factors],
                '敏感度得分': [sens_scores.get(f, 0) for f in factors],
                '回归重要性': [reg_importance.get(f, 0) for f in factors],
                '贡献度得分': [contrib_scores.get(f, 0) for f in factors],
                '综合得分': [final_scores.get(f, 0) for f in factors]
            }, index=factors)

            styled_analysis_df = analysis_df.round(4).style.set_properties(**{
                'color': 'black',
                'font-weight': 'bold'
            })
            st.dataframe(styled_analysis_df)

            # -------------------------------------------------
            # 🎯 最终影响因素优先级排序（修复版）
            # -------------------------------------------------
            st.subheader("2.最终影响因素优先级排序")  # 🎯

            # 一次性拼完整 HTML，避免 Streamlit 自动转义或闭合
            html_parts = [
                '''
                <div style="border:2px solid #007BFF;border-radius:8px;padding:15px;background-color:#f8f9fa;margin:10px 0;">
                '''
            ]

            for idx, (factor, score) in enumerate(ranking.items(), 1):
                html_parts.append(f'''
                <div style="font-size:18px;font-weight:bold;padding:8px;margin:5px 0;
                            background-color:white;border-radius:5px;border-left:3px solid #e9ecef;
                            text-align:center;">
                  <span style="color:#007BFF;">第{idx}位:</span>
                  <span style="color:#28a745;">{factor}</span>
                  （得分：<span style="color:#dc3545;font-weight:bold;">{score:.4f}</span>）
                </div>
                ''')

            html_parts.append('</div>')

            # 整段一次性输出，**唯一**的 unsafe_allow_html=True
            st.markdown(''.join(html_parts), unsafe_allow_html=True)

            # 添加指标说明 - 采用两列布局
            with st.expander("🔍 指标含义说明", expanded=False):
                # 使用两列布局使内容更紧凑
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("""
                    <div style="background-color: #f8f9fa; padding: 10px; margin-bottom: 8px; border-radius: 6px; border-left: 3px solid #007BFF;">
                    <strong>1. 相关性得分 (Correlation Score)</strong><br>
                    <em>业务含义</em>: 衡量因素与OEE之间的线性相关程度，得分越高表示该因素与OEE的相关性越强<br>
                    <em>计算方式</em>: 使用皮尔逊相关系数计算因素与OEE的相关性，然后进行归一化处理<br>
                    <em>计算公式</em>: r = Σ(xi - x̄)(yi - ȳ) / √[Σ(xi - x̄)²Σ(yi - ȳ)²]<br>
                    <em>公式含义</em>: 其中xi为因素值，x̄为因素平均值，yi为OEE值，ȳ为OEE平均值
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("""
                    <div style="background-color: #f8f9fa; padding: 10px; margin-bottom: 8px; border-radius: 6px; border-left: 3px solid #007BFF;">
                    <strong>3. 回归重要性 (Regression Importance)</strong><br>
                    <em>业务含义</em>: 衡量因素在预测OEE中的重要性，得分越高表示该因素对OEE的解释力越强<br>
                    <em>计算方式</em>: 使用线性回归模型，计算各因素系数的绝对值占比<br>
                    <em>计算公式</em>: Importance_j = |βj| / Σ|βk| (k=1 to m)<br>
                    <em>公式含义</em>: 其中βj为第j个因素的回归系数，m为因素总数
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown("""
                    <div style="background-color: #f8f9fa; padding: 10px; margin-bottom: 8px; border-radius: 6px; border-left: 3px solid #007BFF;">
                    <strong>2. 敏感度得分 (Sensitivity Score)</strong><br>
                    <em>业务含义</em>: 衡量因素变化对OEE变化的影响程度，得分越高表示该因素的波动对OEE影响越大<br>
                    <em>计算方式</em>: 计算因素变化率与OEE变化率的比值，反映单位因素变化对OEE的影响<br>
                    <em>计算公式</em>: S = (1/n) × Σ|ΔOEEi / ΔFactori|<br>
                    <em>公式含义</em>: 其中ΔOEEi为第i期OEE变化量，ΔFactori为第i期因素变化量，n为数据期数
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("""
                    <div style="background-color: #f8f9fa; padding: 10px; margin-bottom: 8px; border-radius: 6px; border-left: 3px solid #007BFF;">
                    <strong>4. 贡献度得分 (Contribution Score)</strong><br>
                    <em>业务含义</em>: 衡量因素对OEE的直接贡献程度，得分越高表示该因素对OEE的影响越大<br>
                    <em>计算方式</em>: 通过1%的小幅变动测试，计算因素变化对OEE的影响程度<br>
                    <em>计算公式</em>: C = (1/n) × Σ|(OEE_new,i - OEE_base,i) / OEE_base,i × 100%|<br>
                    <em>公式含义</em>: 其中OEE_new,i为因素变动后的新OEE值，OEE_base,i为基准OEE值，n为数据期数
                    </div>
                    """, unsafe_allow_html=True)

                # 综合得分说明放在下方
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 10px; margin-top: 10px; border-radius: 6px; border-left: 3px solid #007BFF;">
                <strong>5. 综合得分 (Final Score)</strong><br>
                <em>业务含义</em>: 综合考虑多个维度的最终评分，用于确定影响因素的优先级<br>
                <em>权重确定</em>: 基于实践经验设定，相关性和敏感度各占30%（反映统计相关性和实际影响），回归重要性和贡献度各占20%（反映预测能力和直接作用）<br>
                <em>计算方式</em>: 加权平均 = 0.3×相关性得分 + 0.3×敏感度得分 + 0.2×回归重要性 + 0.2×贡献度得分
                </div>
                """, unsafe_allow_html=True)

            # 添加OEE计算公式说明 - 也采用两列布局
            with st.expander("📋 OEE计算公式", expanded=False):
                st.markdown("""
                <div style="background-color: #e8f4fd; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8;">
                <h4>⚙️ OEE (Overall Equipment Effectiveness) 综合设备效率</h4>

                <p><strong>计算公式:</strong></p>
                <p style="font-size: 18px; text-align: center; background-color: white; padding: 10px; border-radius: 5px;">
                OEE = 设备有效利用率 × 性能时间 × 良品率
                </p>
                </div>
                """, unsafe_allow_html=True)

                # 使用两列布局展示各组成部分说明
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("""
                    <div style="background-color: #e8f4fd; padding: 12px; margin-top: 10px; border-radius: 6px; border-left: 3px solid #17a2b8;">
                    <p><strong>各组成部分说明:</strong></p>
                    <ul style="margin-bottom: 0;">
                    <li><strong>设备有效利用率 (Availability)</strong>: 实际运行时间 / 计划运行时间</li>
                    <li><strong>性能时间 (Performance)</strong>: （实际产量×理论单件周期时间） / 实际运行时间</li>
                    <li><strong>良品率 (Quality)</strong>: 合格品数量 / 总生产数量</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown("""
                    <div style="background-color: #e8f4fd; padding: 12px; margin-top: 10px; border-radius: 6px; border-left: 3px solid #17a2b8;">
                    <p><strong>业务意义:</strong></p>
                    <ul style="margin-bottom: 0;">
                    <li>OEE是衡量综合设备效率的核心指标，范围在0-1之间</li>
                    <li>数值越高表示设备效率越高，通常85%以上为优秀水平</li>
                    <li>通过分解OEE可识别设备效率损失的具体环节</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)

            self.ranking = ranking
            return ranking

        def visualization_insights(self):
            st.header("📈 可视化洞察分析")

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # 1. 时间序列趋势图
            st.subheader("1. 时间序列趋势图")
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; border-left: 4px solid #f8f9fa; margin-bottom: 10px; font-size: 13px; line-height: 1.5;">
            🔍 <strong>作用</strong>：展示OEE整体随时间的变化趋势，帮助识别设备效率的长期走势、季节性波动或异常点。
            </div>
            <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; border-left: 4px solid #f8f9fa; font-size: 13px; line-height: 1.5;">
            💡 <strong>业务洞察</strong>：OEE趋势图是判断设备运行健康度的‘心电图’。若OEE持续下降，可能反映设备老化、维护不足或工艺退化；若出现突发性下跌，应结合生产日志排查是否发生重大停机、换型或质量问题。管理者可据此设定预警阈值，实现主动干预。
            </div>
            """, unsafe_allow_html=True)

            # 使用Plotly替代matplotlib
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=self.df['时间'],
                y=self.df['OEE'],
                mode='lines+markers',
                name='OEE',
                line=dict(width=2, color='blue'),
                marker=dict(size=4)
            ))
            fig1.update_layout(
                title={
                    'text': "OEE时间趋势",
                    'font': {'size': 16, 'weight': 'bold'}
                },
                xaxis_title="时间",
                yaxis_title="OEE",
                height=300,
                showlegend=True,
                font=dict(size=12)
            )
            st.plotly_chart(fig1, use_container_width=True)

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # 2. 各因素时间趋势
            st.subheader("2. 各因素时间趋势")
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; border-left: 4px solid #f8f9fa; margin-bottom: 10px; font-size: 13px; line-height: 1.5;">
            🔍 <strong>作用</strong>：分别展示设备有效利用率、性能时间和良品率随时间的变化趋势，帮助识别各OEE组成要素的稳定性与变化模式。
            </div>
            <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; border-left: 4px solid #f8f9fa; font-size: 13px; line-height: 1.5;">
            💡 <strong>业务洞察</strong>：通过观察各因素的时间走势，可判断效率损失的来源是否具有周期性或突发性。例如，若‘设备有效利用率’在某月骤降，可能对应一次重大设备故障；若‘性能时间’持续偏低，说明设备长期未达设计速度，需排查工艺或维护问题；而‘良品率’的波动则可能暴露质量控制薄弱环节。管理者可结合具体业务事件（如换型、维修、原料批次变更）进行根因分析。
            </div>
            """, unsafe_allow_html=True)

            # 使用Plotly创建子图
            from plotly.subplots import make_subplots
            factors = ['设备有效利用率', '性能时间', '良品率']
            colors = ['red', 'blue', 'green']

            fig2 = make_subplots(rows=1, cols=3, subplot_titles=factors)

            for i, (factor, color) in enumerate(zip(factors, colors), 1):
                fig2.add_trace(
                    go.Scatter(
                        x=self.df['时间'],
                        y=self.df[factor],
                        mode='lines+markers',
                        name=factor,
                        line=dict(color=color, width=1),
                        marker=dict(size=2)
                    ),
                    row=1, col=i
                )

            fig2.update_layout(
                height=300,
                showlegend=False,
                font=dict(size=10)
            )

            # 更新子图标题
            for i in range(3):
                fig2.layout.annotations[i].update(font=dict(size=12, weight='bold'))

            st.plotly_chart(fig2, use_container_width=True)

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # 3. 相关性热力图 - 修改为Plotly版本
            st.subheader("3. 相关性热力图")
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; border-left: 4px solid #f8f9fa; margin-bottom: 10px; font-size: 13px; line-height: 1.5;">
            🔍 <strong>作用</strong>：展示OEE与各组成因素之间的线性相关强度。
            </div>
            <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; border-left: 4px solid #f8f9fa; font-size: 13px; line-height: 1.5;">
            💡 <strong>业务洞察</strong>：颜色越红（正相关）或越蓝（负相关），说明该因素对OEE的影响越直接。管理者可据此判断哪些环节的改进能最有效提升整体设备效率。例如，若‘设备有效利用率’与OEE高度正相关，说明减少停机是提效关键。
            </div>
            """, unsafe_allow_html=True)

            # 计算相关性矩阵
            corr_matrix = self.df[['OEE', '设备有效利用率', '性能时间', '良品率']].corr()

            # 使用Plotly创建热力图
            fig3 = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu_r',
                zmid=0,
                text=corr_matrix.round(3).values,
                texttemplate="%{text}",
                textfont={"size": 12},
                hoverinfo="none"
            ))

            fig3.update_layout(
                title={
                    'text': "OEE与各因素相关性热力图",
                    'font': {'size': 16, 'weight': 'bold'}
                },
                xaxis_title="因素",
                yaxis_title="因素",
                height=400,
                width=500,
                font=dict(size=12)
            )

            st.plotly_chart(fig3, use_container_width=True)

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # 4. 散点图矩阵 - 4×4子图版本（修复标题大小和边框问题）
            st.subheader("4. 散点图矩阵")
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; border-left: 4px solid #f8f9fa; margin-bottom: 10px; font-size: 13px; line-height: 1.5;">
            🔍 <strong>作用</strong>：揭示OEE与各因素之间的非线性关系及数据分布形态。
            </div>
            <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; border-left: 4px solid #f8f9fa; font-size: 13px; line-height: 1.5;">
            💡 <strong>业务洞察</strong>：散点图能暴露异常值、聚类或拐点。例如，若'良品率'在高OEE区间突然下降，可能意味着高速生产牺牲了质量——这提示管理者需在效率与质量间寻找平衡点，避免盲目提速。
            </div>
            """, unsafe_allow_html=True)

            # 使用Plotly创建4×4子图布局
            factors = ['OEE', '设备有效利用率', '性能时间', '良品率']

            # 创建4×4子图，设置水平和垂直间距，不添加子图标题
            fig4 = make_subplots(
                rows=4,
                cols=4,
                shared_xaxes=False,
                shared_yaxes=False,
                horizontal_spacing=0.08,  # 水平间距
                vertical_spacing=0.08,  # 垂直间距
                subplot_titles=[''] * 16  # 空标题，避免重叠
            )

            # 为每个子图添加相应的数据
            for i, y_factor in enumerate(factors, 1):
                for j, x_factor in enumerate(factors, 1):
                    row = i
                    col = j

                    if i == j:
                        # 对角线：直方图
                        hist_data = self.df[x_factor]
                        fig4.add_trace(
                            go.Histogram(
                                x=hist_data,
                                nbinsx=15,
                                marker_color='lightblue',
                                opacity=0.7,
                                name=f'{x_factor}分布',
                                showlegend=False
                            ),
                            row=row, col=col
                        )
                    else:
                        # 非对角线：散点图
                        fig4.add_trace(
                            go.Scatter(
                                x=self.df[x_factor],
                                y=self.df[y_factor],
                                mode='markers',
                                marker=dict(
                                    size=6,
                                    color='blue',
                                    opacity=0.6,
                                    line=dict(width=0.5, color='darkblue')
                                ),
                                showlegend=False
                            ),
                            row=row, col=col
                        )

            # 更新布局和样式 - 主标题靠左对齐，字号改为16与其他图一致
            fig4.update_layout(
                title={
                    'text': "OEE与各因素散点图矩阵",
                    'font': {'size': 16, 'weight': 'bold'},  # 从20改为16
                    'x': 0,  # 左对齐
                    'xanchor': 'left'
                },
                height=900,  # 增加高度以适应4×4布局
                showlegend=False,
                paper_bgcolor='white',
                plot_bgcolor='white',
                margin=dict(l=60, r=40, t=80, b=60)  # 增加右侧边距确保最后一列边框可见
            )

            # 为每个子图添加边框、网格和坐标轴标签
            for i in range(1, 5):
                for j in range(1, 5):
                    # 设置X轴标签 - 只在最后一行显示
                    if i == 4:
                        fig4.update_xaxes(
                            title_text=factors[j - 1],
                            title_font=dict(size=10),
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(200,200,200,0.5)',
                            linecolor='black',
                            linewidth=1,
                            mirror=True,
                            showline=True,  # 确保显示线条
                            row=i, col=j
                        )
                    else:
                        fig4.update_xaxes(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(200,200,200,0.5)',
                            linecolor='black',
                            linewidth=1,
                            mirror=True,
                            showline=True,  # 确保显示线条
                            showticklabels=True,  # 确保显示刻度标签
                            row=i, col=j
                        )

                    # 设置Y轴标签 - 只在第一列显示
                    if j == 1:
                        fig4.update_yaxes(
                            title_text=factors[i - 1],
                            title_font=dict(size=10),
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(200,200,200,0.5)',
                            linecolor='black',
                            linewidth=1,
                            mirror=True,
                            showline=True,  # 确保显示线条
                            row=i, col=j
                        )
                    else:
                        fig4.update_yaxes(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(200,200,200,0.5)',
                            linecolor='black',
                            linewidth=1,
                            mirror=True,
                            showline=True,  # 确保显示线条
                            showticklabels=True,  # 确保显示刻度标签
                            row=i, col=j
                        )

            # 特别确保最后一列的右侧边框显示
            for i in range(1, 5):
                fig4.update_yaxes(
                    showline=True,
                    linecolor='black',
                    linewidth=1,
                    mirror=True,
                    row=i, col=4
                )

            st.plotly_chart(fig4, use_container_width=True)

            # 计算月度波动数据（在使用前定义）
            monthly_std = self.df[['OEE', '设备有效利用率', '性能时间', '良品率']].std()

            # 使用两列布局来更好地展示饼图和柱状图
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("5. 影响因素贡献度")
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; border-left: 4px solid #f8f9fa; margin-bottom: 10px; font-size: 13px; line-height: 1.5;">
                🔍 <strong>作用</strong>：量化各因素对OEE变动的相对贡献大小。
                </div>
                <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; border-left: 4px solid #f8f9fa; font-size: 13px; line-height: 1.5;">
                💡 <strong>业务洞察</strong>：贡献度高的因素应优先投入资源改进。例如，若‘性能时间’贡献最大，说明设备运行速度或小停机是瓶颈，建议开展TPM（全员生产维护）或优化工艺参数，而非盲目增加设备数量。
                </div>
                """, unsafe_allow_html=True)
                if hasattr(self, 'ranking') and len(self.ranking) > 0:
                    ranking_values = [max(0.01, v) if not np.isnan(v) else 0.01 for v in self.ranking.values]
                    ranking_labels = self.ranking.index

                    # 使用Plotly饼图
                    fig5 = go.Figure(data=[go.Pie(
                        labels=ranking_labels,
                        values=ranking_values,
                        hole=0.3,
                        textinfo='percent+label',
                        insidetextorientation='radial'
                    )])
                    fig5.update_layout(
                        title={
                            'text': "贡献度分布",
                            'font': {'size': 14, 'weight': 'bold'}
                        },
                        height=300
                    )
                    st.plotly_chart(fig5, use_container_width=True)

            with col2:
                st.subheader("6. OEE组成分析")
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; border-left: 4px solid #f8f9fa; margin-bottom: 10px; font-size: 13px; line-height: 1.5;">
                🔍 <strong>作用</strong>：分解OEE为三大损失（可用性、性能、质量），直观展示效率损失结构。
                </div>
                <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; border-left: 4px solid #f8f9fa; font-size: 13px; line-height: 1.5;">
                💡 <strong>业务洞察</strong>：饼图揭示‘看不见的浪费’。例如，若‘性能损失’占比最高，说明设备虽在运行但未达理想速度——这往往是改善空间最大的环节。管理者应聚焦于此，而非仅关注设备是否开机。
                </div>
                """, unsafe_allow_html=True)
                avg_utilization = self.df['设备有效利用率'].mean()
                avg_performance = self.df['性能时间'].mean()
                avg_quality = self.df['良品率'].mean()
                avg_oee = self.df['OEE'].mean()
                theoretical_max = 1.0
                utilization_loss = theoretical_max - avg_utilization
                performance_loss = avg_utilization - avg_utilization * avg_performance
                quality_loss = avg_utilization * avg_performance - avg_oee
                components = [avg_oee, quality_loss, performance_loss, utilization_loss]
                labels = ['OEE', '质量', '性能', '可用性']
                colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']

                # 使用Plotly饼图
                fig6 = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=components,
                    hole=0.3,
                    marker_colors=colors,
                    textinfo='percent+label'
                )])
                fig6.update_layout(
                    title={
                        'text': "OEE组成(平均)",
                        'font': {'size': 14, 'weight': 'bold'}
                    },
                    height=300
                )
                st.plotly_chart(fig6, use_container_width=True)

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # 7. 各指标波动程度
            st.subheader("7. 各指标波动程度")
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; border-left: 4px solid #f8f9fa; margin-bottom: 10px; font-size: 13px; line-height: 1.5;">
            🔍 <strong>作用</strong>：通过标准差衡量各指标的稳定性，识别波动最大的环节。
            </div>
            <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; border-left: 4px solid #f8f9fa; font-size: 13px; line-height: 1.5;">
            💡 <strong>业务洞察</strong>：波动大的指标意味着过程不稳定，是质量或交付风险的源头。例如，若‘良品率’波动剧烈，可能反映来料不稳或操作不规范，建议加强过程控制（SPC）和标准化作业，而非仅追责操作员。
            </div>
            """, unsafe_allow_html=True)

            # 使用Plotly柱状图
            fig7 = go.Figure(data=[go.Bar(
                x=monthly_std.index,
                y=monthly_std.values,
                marker_color=['blue', 'red', 'green', 'orange'],
                text=monthly_std.round(3).values,
                textposition='auto',
            )])
            fig7.update_layout(
                title={
                    'text': "月度波动程度(标准差)",
                    'font': {'size': 16, 'weight': 'bold'}
                },
                xaxis_title="指标",
                yaxis_title="标准差",
                height=300,
                font=dict(size=12)
            )
            st.plotly_chart(fig7, use_container_width=True)

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # 关键统计指标
            st.subheader("📊 关键统计指标")
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; border-left: 4px solid #f8f9fa; margin-bottom: 10px; font-size: 13px; line-height: 1.5;">
            🔍 <strong>作用</strong>：提供OEE及其三大组成要素（设备有效利用率、性能时间、良品率）的核心统计摘要，包括均值、标准差、最小/最大值、四分位数等，用于快速评估数据分布特征与整体水平。
            </div>
            <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; border-left: 4px solid #f8f9fa; font-size: 13px; line-height: 1.5;">
            💡 <strong>业务洞察</strong>：统计指标是判断设备运行稳定性和效率基线的‘体检报告’。例如，若OEE均值高但标准差大，说明设备效率波动剧烈，可能存在偶发性故障或操作不规范；若良品率的最小值远低于均值，提示某些批次存在严重质量问题。管理者可据此设定控制限、识别异常时段，并为持续改进提供基准参考。
            </div>
            """, unsafe_allow_html=True)
            stats_df = self.df[['OEE', '设备有效利用率', '性能时间', '良品率']].describe()
            styled_stats_df = stats_df.round(4).style.set_properties(**{
                'color': 'black',
                'font-weight': 'bold'
            })
            st.dataframe(styled_stats_df)

        def actionable_recommendations(self):
            if not hasattr(self, 'ranking'):
                self.quantitative_analysis()
            valid_ranking = self.ranking.dropna()
            if len(valid_ranking) == 0:
                st.warning("无法确定影响因素优先级，所有得分均为NaN")
                return

            top_factor = valid_ranking.index[0]
            second_factor = valid_ranking.index[1] if len(valid_ranking) > 1 else None

            recommendations = {
                '设备有效利用率': [
                    "减少计划外停机时间",
                    "优化设备换型流程，缩短换型时间",
                    "加强预防性维护，减少故障停机",
                    "改进物料供应系统，避免待料停机",
                    "制定标准作业程序，减少操作失误"
                ],
                '性能时间': [
                    "识别并消除设备小停机",
                    "优化设备运行参数，提高运行速度",
                    "减少设备空转和等待时间",
                    "改进生产工艺流程",
                    "加强操作人员技能培训"
                ],
                '良品率': [
                    "加强来料质量控制",
                    "优化工艺参数设置",
                    "改进设备精度和稳定性",
                    "加强过程质量监控",
                    "实施根本原因分析，减少质量波动"
                ]
            }

            # ================ 一、可执行改进建议 ================
            st.markdown("##  一、可执行改进建议")
            # --- 1.1 主要改进方向 ---
            st.markdown("### 🎯 1.1 主要改进方向")
            st.info(f"重点关注 **{top_factor}**，该因素对OEE影响最大。")

            # --- 1.2 次要改进方向（如有）---
            if second_factor:
                st.markdown("### 🔍 1.2 次要改进方向")
                st.info(f"同时关注 **{second_factor}**，以协同提升整体效率。")

            # --- 1.3 具体改进措施 ---
            st.markdown("### 📋 1.3 具体改进措施")

            if second_factor:
                col_main, col_aux = st.columns(2)
                with col_main:
                    st.markdown(f"**针对【{top_factor}】的主要措施：**")
                    for rec in recommendations[top_factor]:
                        st.markdown(f"✔️ {rec}")
                with col_aux:
                    st.markdown(f"**针对【{second_factor}】的辅助措施：**")
                    for rec in recommendations[second_factor][:3]:
                        st.markdown(f"✔️ {rec}")
            else:
                st.markdown(f"**针对【{top_factor}】的具体措施：**")
                for rec in recommendations[top_factor]:
                    st.markdown(f"✔️ {rec}")

            st.markdown("---")

            # ================ 二、基于数据分析的具体问题 ================
            st.markdown("## 二、基于数据分析的具体问题")

            worst_month_idx = self.df['OEE'].idxmin()
            worst_month = self.df.loc[worst_month_idx, '时间'].strftime('%Y-%m')
            worst_oee = self.df.loc[worst_month_idx, 'OEE']
            worst_util = self.df.loc[worst_month_idx, '设备有效利用率']
            worst_perf = self.df.loc[worst_month_idx, '性能时间']
            worst_qual = self.df.loc[worst_month_idx, '良品率']

            avg_util = self.df['设备有效利用率'].mean()
            avg_perf = self.df['性能时间'].mean()
            avg_qual = self.df['良品率'].mean()

            problem_data = {
                "指标": ["设备有效利用率", "性能时间", "良品率"],
                "最差月值": [worst_util, worst_perf, worst_qual],
                "平均值": [avg_util, avg_perf, avg_qual],
                "差距": [
                    worst_util - avg_util,
                    worst_perf - avg_perf,
                    worst_qual - avg_qual
                ]
            }
            problem_df = pd.DataFrame(problem_data)
            problem_df["差距"] = problem_df["差距"].round(4)
            problem_df["最差月值"] = problem_df["最差月值"].round(4)
            problem_df["平均值"] = problem_df["平均值"].round(4)

            st.markdown(f"**表现最差月份**：{worst_month}（OEE: {worst_oee:.3f}）")

            # === 使用 HTML 表格确保加粗加黑 ===
            table_html = "<table style='width:100%; border-collapse: collapse; font-weight: bold; color: #000; font-size: 15px;'>"
            table_html += "<thead><tr>"
            for col in problem_df.columns:
                table_html += f"<th style='border: 1px solid #ccc; padding: 10px; background-color: #f0f8ff; text-align: center;'>{col}</th>"
            table_html += "</tr></thead><tbody>"
            for _, row in problem_df.iterrows():
                table_html += "<tr>"
                for val in row:
                    table_html += f"<td style='border: 1px solid #ccc; padding: 10px; text-align: center;'>{val}</td>"
                table_html += "</tr>"
            table_html += "</tbody></table>"

            st.markdown(table_html, unsafe_allow_html=True)

            # --- 2.1 波动最大的因素（恢复你丢失的内容）---
            most_volatile = self.df[['设备有效利用率', '性能时间', '良品率']].std().idxmax()
            volatility = self.df[most_volatile].std()
            st.markdown(f"**波动最大的因素**: {most_volatile}（标准差: {volatility:.4f}）——建议加强过程稳定性控制。")

            st.markdown("---")

            # ================ 三、特殊发现 ================
            st.markdown("## 三、特殊发现")

            findings = []

            # 检查良品率是否几乎不变
            if self.df['良品率'].nunique() <= 2 and self.df['良品率'].std() < 0.001:
                findings.append("• 良品率几乎恒定（如始终为1.0000），可能存在数据采集或定义问题，建议核查质量数据真实性。")

            # 检查性能时间波动是否过大
            perf_min = self.df['性能时间'].min()
            perf_max = self.df['性能时间'].max()
            if perf_min > 0 and perf_max / perf_min > 10:
                findings.append(
                    f"• 性能时间波动极大（{perf_min:.4f} → {perf_max:.4f}），可能反映设备运行不稳定或存在异常数据点。")

            # 检查是否存在某月OEE骤降但无对应因素下降（逻辑矛盾）
            oee_drop = self.df['OEE'].pct_change().min()
            if oee_drop < -0.3:  # 单月下降超30%
                drop_idx = self.df['OEE'].pct_change().idxmin()
                drop_month = self.df.loc[drop_idx, '时间'].strftime('%Y-%m')
                findings.append(
                    f"• OEE在 {drop_month} 出现断崖式下跌（降幅 >30%），建议结合生产日志排查重大停机或质量问题。")

            if findings:
                for f in findings:
                    st.warning(f)
            else:
                st.success("✅ 未发现明显异常或特殊模式。")

        def improved_trend_prediction(self, future_periods=6):
            st.header("🔮 OEE趋势预测")
            try:
                X_time = self.df[['月份序号']]
                y_oee = self.df['OEE']
                linear_model = LinearRegression()
                linear_model.fit(X_time, y_oee)
                future_months = np.array(range(len(self.df) + 1, len(self.df) + future_periods + 1)).reshape(-1, 1)
                predictions = linear_model.predict(future_months)
                future_dates = pd.date_range(
                    start=self.df['时间'].iloc[-1] + pd.DateOffset(months=1),
                    periods=future_periods,
                    freq='M'
                )
                forecast_df = pd.DataFrame({
                    '预测月份': future_dates.strftime('%Y-%m'),
                    '预测OEE': predictions,
                    '预测下限': predictions - 0.15,
                    '预测上限': predictions + 0.15,
                    '趋势': ['平稳' for _ in range(len(predictions))]
                })
                st.subheader(f"未来{future_periods}个月OEE预测结果")
                styled_forecast_df = forecast_df.round(4).style.set_properties(**{
                    'color': 'black',
                    'font-weight': 'bold'
                })
                st.dataframe(styled_forecast_df)

                # 使用Plotly替代matplotlib - 修复错误
                fig = go.Figure()

                # 历史数据
                fig.add_trace(go.Scatter(
                    x=self.df['时间'],
                    y=self.df['OEE'],
                    mode='lines+markers',
                    name='历史OEE',
                    line=dict(width=2, color='blue'),
                    marker=dict(size=4)
                ))

                # 预测数据
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=forecast_df['预测OEE'],
                    mode='lines+markers',
                    name='预测OEE',
                    line=dict(width=2, color='red', dash='dash'),
                    marker=dict(size=4)
                ))

                # 预测区间 - 修复这里的关键错误
                # 将DatetimeIndex转换为Series以便concat
                future_dates_series = pd.Series(future_dates)
                future_dates_reversed = pd.Series(future_dates[::-1])

                # 创建预测区间的x和y数据
                confidence_x = pd.concat([future_dates_series, future_dates_reversed])
                confidence_y = pd.concat([
                    forecast_df['预测上限'],
                    forecast_df['预测下限'][::-1]
                ])

                fig.add_trace(go.Scatter(
                    x=confidence_x,
                    y=confidence_y,
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='预测区间',
                    showlegend=True
                ))

                fig.update_layout(
                    title={
                        'text': "OEE历史趋势与预测",
                        'font': {'size': 16, 'weight': 'bold'}
                    },
                    xaxis_title="时间",
                    yaxis_title="OEE",
                    height=400,
                    showlegend=True,
                    font=dict(size=12)
                )
                st.plotly_chart(fig, use_container_width=True)
                return forecast_df
            except Exception as e:
                st.error(f"预测失败: {e}")
                return None

    # Streamlit应用主函数
    st.set_page_config(
        page_title="OEE设备效率分析系统",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    .stApp header { visibility: visible !important; display: block !important; }
    [data-testid="stSidebar"] { background-color: #f0f2f6; padding: 1rem !important; height: 100vh; overflow-y: auto !important; }
    .stMarkdown, .stText, .stDataFrame, .stCode { line-height: 1.4 !important; font-family: "Source Sans Pro", sans-serif !important; }
    .stApp { overflow-y: auto !important; overflow-x: hidden !important; height: 100vh !important; }
    .main .block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; max-width: 100% !important; min-height: 100vh !important; }
    .stPyplot { max-width: 100% !important; height: auto !important; }
    .stDataFrame { width: 100% !important; overflow-x: auto !important; }
    [data-testid="stMainBlockContainer"] { overflow-y: auto !important; height: calc(100vh - 6rem) !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px !important; }
    .stTabs [data-baseweb="tab"] { height: 35px !important; border-radius: 4px 4px 0px 0px !important; padding: 0px 16px !important; }
    .element-container { margin-bottom: 1rem !important; }
    ::-webkit-scrollbar { width: 8px !important; height: 8px !important; }
    ::-webkit-scrollbar-track { background: #f1f1f1 !important; border-radius: 4px !important; }
    ::-webkit-scrollbar-thumb { background: #c1c1c1 !important; border-radius: 4px !important; }
    ::-webkit-scrollbar-thumb:hover { background: #a8a8a8 !important; }
    .sidebar .sidebar-content { padding: 1rem !important; height: 100% !important; overflow-y: auto !important; }
    p, div, span { line-height: 1.5 !important; }
    section[data-testid="stSidebar"] > div:first-child { height: 100vh !important; overflow-y: auto !important; }
    .block-container { padding-bottom: 5rem !important; }
    .tableScroll { -webkit-overflow-scrolling: touch !important; overflow-x: auto !important; overflow-y: auto !important; max-height: 400px !important; }
    .recommendation-content { font-size: 0.95rem !important; font-weight: normal !important; }
    .stDataFrame th, .stDataFrame td { color: black !important; font-weight: bold !important; }
    /*h1 { font-size: 2rem !important; }*/
    h2 { font-size: 1.4rem !important; }
    h3 { font-size: 1.2rem !important; }
    </style>
    """, unsafe_allow_html=True)

    st.title("📊 OEE设备效率分析系统")
    st.markdown("---")

    with st.sidebar:
        st.header("数据上传与设置")
        uploaded_file = st.file_uploader(
            "上传OEE数据Excel文件",
            type=['xlsx'],
            help="请上传包含月份、设备有效利用率、性能时间、良品率、OEE的Excel文件"
        )
        future_periods = st.slider(
            "预测未来月份数",
            min_value=3,
            max_value=12,
            value=6,
            help="选择要预测的未来月份数量"
        )
        analyze_button = st.button("开始分析", type="primary", use_container_width=True)
        use_sample_data = st.checkbox("使用示例数据", value=False)
        st.markdown("<br>", unsafe_allow_html=True)

    if use_sample_data:
        np.random.seed(42)
        months = 24
        utilization = np.random.normal(0.85, 0.04, months)
        performance = np.random.normal(0.90, 0.03, months)
        quality = np.random.normal(0.95, 0.02, months)
        utilization = np.clip(utilization, 0.7, 0.98)
        performance = np.clip(performance, 0.8, 0.98)
        quality = np.clip(quality, 0.9, 0.99)
        oee = utilization * performance * quality
        sample_data = pd.DataFrame({
            '月份': pd.date_range(start='2023-01', periods=months, freq='M'),
            '设备有效利用率': utilization,
            '性能时间': performance,
            '良品率': quality,
            'OEE': oee
        })
        st.sidebar.success("已加载示例数据")

    if uploaded_file is not None or use_sample_data:
        if analyze_button:
            with st.spinner('正在分析数据，请稍候...'):
                try:
                    if use_sample_data:
                        data = sample_data
                    else:
                        data = pd.read_excel(uploaded_file)
                    analyzer = OEEAnalyzer(data)
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "📈 定量分析",
                        "📊 可视化洞察",
                        "💡 改进建议",
                        "🔮 趋势预测"
                    ])
                    with tab1:
                        analyzer.quantitative_analysis()
                    with tab2:
                        analyzer.visualization_insights()
                    with tab3:
                        analyzer.actionable_recommendations()
                    with tab4:
                        analyzer.improved_trend_prediction(future_periods)
                    st.success("✅ 分析完成！")
                    st.balloons()
                except Exception as e:
                    st.error(f"分析过程中出现错误: {str(e)}")
                    st.info("请检查数据格式是否正确，确保包含以下列：月份、设备有效利用率、性能时间、良品率、OEE")
        else:
            if use_sample_data:
                data = sample_data
            else:
                data = pd.read_excel(uploaded_file)
            st.subheader("数据预览")
            styled_data_preview = data.head(8).style.set_properties(**{
                'color': 'black',
                'font-weight': 'bold'
            })
            st.dataframe(styled_data_preview)
            st.info("点击侧边栏的『开始分析』按钮进行完整分析")
    else:
        st.subheader("使用说明")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("""
            ### 📋 数据格式要求
            - Excel文件格式 (.xlsx)
            - 必须包含以下列：
              - **月份** (日期格式)
              - **设备有效利用率** (0-1之间的小数)
              - **性能时间** (0-1之间的小数)
              - **良品率** (0-1之间的小数)
              - **OEE** (0-1之间的小数)
            """)
        with col2:
            st.markdown("""
            ### 🎯 分析内容
            - **定量分析**: 确定影响OEE的关键因素
            - **可视化洞察**: 多维度图表分析
            - **改进建议**: 基于数据的可操作性建议
            - **趋势预测**: 未来OEE趋势预测
            """)
        st.markdown("---")
        st.info("请在左侧边栏上传Excel文件或选择使用示例数据")


def run_production_system():
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from datetime import datetime, timedelta
    import warnings
    import os
    from sklearn.linear_model import LinearRegression
    from prophet import Prophet
    import streamlit as st

    warnings.filterwarnings("ignore")

    @st.cache_data
    def load_real_data():
        base_dir = os.path.dirname(os.path.abspath(__file__))
        try:
            equipment_df = pd.read_csv(os.path.join(base_dir, "equipment_base.csv"), encoding='utf-8')
            status_df = pd.read_csv(os.path.join(base_dir, "equipment_status_daily.csv"), parse_dates=['日期'],
                                    encoding='utf-8')
            efficiency_df = pd.read_csv(os.path.join(base_dir, "equipment_efficiency_daily.csv"), parse_dates=['日期'],
                                        encoding='utf-8')
            buffer_df = pd.read_csv(os.path.join(base_dir, "buffer_inventory_daily.csv"), parse_dates=['日期'],
                                    encoding='utf-8')
            prod_df = pd.read_csv(os.path.join(base_dir, "production_daily.csv"), parse_dates=['日期'], encoding='utf-8')

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

    class ProductionDashboard:
        def __init__(self, equipment_df, status_df, efficiency_df, buffer_df, prod_df):
            self.equipment_df = equipment_df
            self.status_df = status_df
            self.efficiency_df = efficiency_df
            self.buffer_df = buffer_df
            self.prod_df = prod_df
            self.optimization_engine = OptimizationEngine(equipment_df)
            self.filler_ids = equipment_df[equipment_df['工序类型'] == '灌装']['设备ID'].tolist()
            self.packer_ids = equipment_df[equipment_df['工序类型'] == '包装']['设备ID'].tolist()
            self.selected_date = None
            self.date_range = None
            self.SAFE_BUFFER = 2880

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
            prod_row = self.prod_df[self.prod_df['日期'] == target_date]
            if prod_row.empty:
                st.error(f"⚠️ 日期 {self.selected_date} 在 production_daily.csv 中无数据")
                st.stop()
            prod_row = prod_row.iloc[0]
            plan_yield = prod_row['计划产量(瓶)']
            actual_yield = prod_row['总产量(瓶)']
            gap_ratio = max(0.0, (plan_yield - actual_yield) / plan_yield) if plan_yield > 0 else 0.0
            eff_row = self.efficiency_df[self.efficiency_df['日期'] == target_date]
            if eff_row.empty:
                oee = utilization = 0.0
            else:
                eff_row = eff_row.iloc[0]
                oee = eff_row['综合效率(OEE)']
                utilization = eff_row['产能利用率']
            filler_output = self.status_df[
                (self.status_df['日期'] == target_date) &
                (self.status_df['设备ID'].isin(self.filler_ids))
                ]['当日产量(瓶)'].sum()
            packer_output = self.status_df[
                (self.status_df['日期'] == target_date) &
                (self.status_df['设备ID'].isin(self.packer_ids))
                ]['当日产量(瓶)'].sum()
            balance_ratio = filler_output / (packer_output + 1e-8)
            buffer_row = self.buffer_df[self.buffer_df['日期'] == target_date]
            buffer_level = buffer_row['期末数量(盘)'].sum() if not buffer_row.empty else 0
            buffer_risk_score = min(buffer_level / self.SAFE_BUFFER, 1.2)
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
            # === 新增：增强 Plotly 标题 ===
            fig.update_layout(
                title={
                    'text': "车间日产量趋势",
                    'font': {'size': 18, 'weight': 'bold'}
                }
            )
            st.plotly_chart(fig, use_container_width=True)

        def _plot_prophet_prediction(self):
            start, end = self.date_range
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
            df = self.prod_df[(self.prod_df['日期'] >= start_ts) & (self.prod_df['日期'] <= end_ts)].copy()
            if df.empty:
                st.warning("所选日期范围内无生产数据，无法预测")
                return
            pred_days = st.session_state.get("pred_days", 3)
            prophet_df = df[['日期', '总产量(瓶)']].rename(columns={'日期': 'ds', '总产量(瓶)': 'y'})
            prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
            model = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=False,
                interval_width=0.95
            )
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=pred_days, freq='D')
            forecast = model.predict(future)
            hist_df = prophet_df.copy()
            hist_df['类型'] = '历史'
            pred_df = forecast[['ds', 'yhat']].tail(pred_days).copy()
            pred_df = pred_df.rename(columns={'ds': '日期', 'yhat': '总产量(瓶)'})
            pred_df['总产量(瓶)'] = np.maximum(pred_df['总产量(瓶)'], 0)
            pred_df['类型'] = '预测'
            hist_df = hist_df.rename(columns={'ds': '日期', 'y': '总产量(瓶)'})
            plot_df = pd.concat([hist_df, pred_df], ignore_index=True)
            fig = px.line(
                plot_df,
                x='日期',
                y='总产量(瓶)',
                color='类型',
                title="日产量预测趋势",
                line_dash='类型',
                labels={'总产量(瓶)': '产量(瓶)'}
            )
            fig.update_traces(mode='lines+markers')
            # === 新增：增强 Plotly 标题 ===
            fig.update_layout(
                title={
                    'text': "日产量预测趋势",
                    'font': {'size': 18, 'weight': 'bold'}
                }
            )
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

            # 创建折线图 - 完全匹配产量趋势图的样式
            fig = go.Figure()

            # 添加主趋势线 - 使用与其他图表一致的蓝色
            fig.add_trace(go.Scatter(
                x=df['日期'],
                y=df['库存占比'],
                mode='lines+markers',
                name='库存占比',
                line=dict(
                    width=2,
                    color='#1f77b4',  # 修改为与其他图表一致的蓝色
                ),
                marker=dict(
                    size=4,
                    color='#1f77b4'  # 标记点也使用相同颜色
                ),
                hovertemplate=(
                        '<b>日期</b>: %{x|%Y-%m-%d}<br>' +
                        '<b>库存占比</b>: %{y:.2%}<br>' +
                        '<b>实际库存</b>: ' + df['期末数量(盘)'].astype(str) + ' 盘<extra></extra>'
                )
            ))

            # 添加预警线
            high_thresh = st.session_state.get("high_thresh", 0.8)
            low_thresh = st.session_state.get("low_thresh", 0.2)

            fig.add_hline(
                y=high_thresh,
                line_dash="dash",
                line_color="red",
                line_width=2,
                annotation_text=f"高预警 {high_thresh:.0%}",
                annotation_position="bottom right"
            )

            fig.add_hline(
                y=low_thresh,
                line_dash="dash",
                line_color="green",
                line_width=2,
                annotation_text=f"低预警 {low_thresh:.0%}",
                annotation_position="top right"
            )

            # 布局 - 完全匹配其他图表的标题样式
            fig.update_layout(
                title={
                    'text': "缓冲区库存占比趋势",
                    'font': {'size': 18, 'weight': 'bold'}
                },
                xaxis_title="日期",
                yaxis_title="库存占比",
                height=400,
                showlegend=True,
                font=dict(size=12)
            )

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
                 /* 直接设置 expander 的 summary 样式 */
                div[data-testid="stExpander"] > details > summary {
                    font-size: 10px !important;
                    font-weight: normal !important;
                    color: #444 !important;
                }
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

            col_main, col_side = st.columns([2, 1])
            with col_main:
                self._show_production_overview(current_state)

                # === 新增：整体业务价值说明（可收放） ===
                with st.expander("系统整体业务说明", expanded=False):
                    st.markdown("""
                                **本系统构建了“监控 → 预测 → 调节”的智能闭环，实现三大核心业务价值：**

                                - **实时监控**：动态追踪产量、效率、缓冲库存等关键指标，确保生产透明可控；
                                - **智能预测**：基于历史数据预测未来产能趋势，提前识别交付风险；
                                - **自动调节**：通过产线平衡分析与瓶颈诊断，提供可执行的优化建议，保障计划达成。

                                该闭环机制显著提升交付准时率、降低库存积压、优化设备利用率，驱动精益生产。
                                """)

                # === 图1：日产量趋势 + 说明卡片 ===
                self._plot_production_trends()
                with st.expander("日产量趋势图作用与业务洞察", expanded=False):#📊
                    st.markdown("""
                    **作用**：直观展示实际产量与计划产量的每日对比，识别波动与趋势。  
                    **业务洞察**：帮助管理者快速判断产能达成情况，及时干预偏离计划的生产日，保障订单交付稳定性。
                    """)

                # === 图2：产量预测 + 说明卡片 ===
                self._plot_prophet_prediction()
                with st.expander("产量预测图作用与业务洞察", expanded=False):#🔮
                    st.markdown("""
                    **作用**：基于历史数据预测未来3-14天产量走势，量化不确定性区间。  
                    **业务洞察**：提前预警潜在产能缺口，支持排产、人力与物料的前瞻性调度，降低交付风险。
                    """)

            with col_side:
                self._show_alerts_warnings(report['risks'])
                self._show_optimization_recommendations(report['recommendations'], current_state)

            st.markdown("---")

            # === 图3：缓冲区分析 + 说明卡片 ===
            self._show_buffer_analysis(current_state)
            with st.expander("缓冲区库存占比图作用与业务洞察", expanded=False):#📦
                st.markdown("""
                **作用**：监控缓冲区库存水平，识别积压或断料风险。  
                **业务洞察**：通过动态平衡灌装与包装节拍，减少在制品堆积，提升产线协同效率，避免非计划停机。
                """)

    equipment_df, status_df, efficiency_df, buffer_df, prod_df = load_real_data()
    dashboard = ProductionDashboard(equipment_df, status_df, efficiency_df, buffer_df, prod_df)
    dashboard.run_dashboard()


# ==============================
# 主入口：系统选择器
# ==============================
def main():
    st.set_page_config(
        page_title="OEE与生产调度双系统",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with st.sidebar:
        system_choice = st.radio(
            "选择系统",
            ("OEE设备效率分析系统", "智能生产调度与产能优化系统"),
            index=0
        )
        st.markdown("---")
        #st.markdown("💡 使用上方切换系统")

    if system_choice == "OEE设备效率分析系统":
        run_oee_system()
    else:
        run_production_system()


if __name__ == "__main__":
    main()



