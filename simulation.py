import pandas as pd
import numpy as np
import random
from datetime import datetime


def generate_equipment_reports(production_daily_df, equipment_df):
    """
    根据生产日报表和设备基础信息表生成设备状态日报表和设备效能日报表

    参数:
    production_daily_df (DataFrame): 生产日报表，包含'日期', '工序类型', '总产量(瓶)'列
    equipment_df (DataFrame): 设备基础信息表，包含'设备ID', '理论速度(瓶/天)', '效率', '可用率'列

    返回:
    equipment_status_df (DataFrame): 设备状态日报表
    equipment_efficiency_df (DataFrame): 设备效能日报表
    """
    # 验证输入数据
    required_production_cols = ['日期', '工序类型', '总产量(瓶)']
    required_equipment_cols = ['设备ID', '理论速度(瓶/天)', '效率', '可用率']

    if not all(col in production_daily_df.columns for col in required_production_cols):
        raise ValueError(f"生产日报表缺少必要列: {required_production_cols}")

    if not all(col in equipment_df.columns for col in required_equipment_cols):
        raise ValueError(f"设备基础信息表缺少必要列: {required_equipment_cols}")

    # 1. 从生产日报表中提取每日各工序总产量
    daily_production = production_daily_df.groupby(['日期', '工序类型']).agg(
        {'总产量(瓶)': 'sum'}
    ).reset_index()

    # 2. 初始化设备状态日报表记录
    equipment_status_records = []

    # 3. 遍历每一天和每个工序类型
    for _, row in daily_production.iterrows():
        date = row['日期']
        process_type = row['工序类型']
        total_output = row['总产量(瓶)']

        # 确定对应设备类型（根据工序类型匹配设备ID前缀）
        if process_type == '灌装':
            prefix = 'FILLER'
        elif process_type == '灭菌':
            prefix = 'STERILIZER'
        elif process_type == '包装':
            prefix = 'PACKAGER'
        else:
            continue  # 跳过不支持的工序类型

        # 从设备基础信息表中筛选相关设备
        equipment_df_filtered = equipment_df[equipment_df['设备ID'].str.startswith(prefix)]

        if equipment_df_filtered.empty:
            continue  # 无相关设备，跳过

        # 4. 计算设备权重（理论速度 × 效率）
        equipment_df_filtered = equipment_df_filtered.copy()
        equipment_df_filtered['weight'] = equipment_df_filtered['理论速度(瓶/天)'] * equipment_df_filtered['效率']
        total_weight = equipment_df_filtered['weight'].sum()

        # 5. 为每台设备分配产量
        for _, equipment_row in equipment_df_filtered.iterrows():
            equipment_id = equipment_row['设备ID']
            weight = equipment_row['weight']

            # 分配产量 = 总产量 × (设备权重 / 总权重)
            output = int(total_output * (weight / total_weight)) if total_weight > 0 else 0

            # 6. 模拟设备运行状态（基于可用率）
            available_rate = equipment_row['可用率']
            status = '运行' if random.random() < available_rate else '停机'

            # 7. 计算运行/停机时长
            if status == '运行':
                # 运行时长 = 16小时 * 可用率（限制在12-16小时）
                run_hours = max(12, min(16, int(16 * available_rate)))
                downtime_hours = 16 - run_hours
            else:
                run_hours = 0
                downtime_hours = 16  # 计划工作16小时，全部停机

            # 8. 生成故障信息
            if status == '停机':
                fault_types = ['机械', '电气', '软件', '液压']
                fault_type = random.choice(fault_types)
                fault_desc = f'{fault_type}故障'
            else:
                fault_type = '无'
                fault_desc = '正常'

            # 9. 添加到设备状态记录
            equipment_status_records.append({
                '日期': date,
                '设备ID': equipment_id,
                '当日产量(瓶)': output,
                '运行时长(小时)': run_hours,
                '停机时长(小时)': downtime_hours,
                '状态': status,
                '故障类型': fault_type,
                '故障描述': fault_desc
            })

    # 10. 生成设备状态日报表
    equipment_status_df = pd.DataFrame(equipment_status_records)

    # 11. 生成设备效能日报表
    equipment_efficiency_records = []

    # 12. 遍历设备状态日报表中的每条记录
    for _, status_row in equipment_status_df.iterrows():
        date = status_row['日期']
        equipment_id = status_row['设备ID']
        actual_output = status_row['当日产量(瓶)']

        # 从设备基础信息表获取设备参数
        equipment_info = equipment_df[equipment_df['设备ID'] == equipment_id]
        if equipment_info.empty:
            continue

        theoretical_capacity = equipment_info['理论速度(瓶/天)'].values[0]
        efficiency = equipment_info['效率'].values[0]

        # 13. 计算效能指标（严格遵循知识库[2][8]的OEE公式）
        # 时间开动率 = 实际运行时间 / 计划运行时间
        run_hours = status_row['运行时长(小时)']
        availability = run_hours / 16  # 计划运行时间=16小时

        # 产能效率 = 实际产量 / 理论产量（确保不超过100%）
        performance = min(1.0, actual_output / theoretical_capacity) if theoretical_capacity > 0 else 0

        # 良率 = 设备效率
        quality = efficiency

        # OEE = 时间开动率 × 产能效率 × 良率
        oee = availability * performance * quality

        # 14. 模拟故障数据
        fault_count = 0
        fault_hours = 0
        if actual_output == 0:
            fault_count = random.randint(1, 3)
            fault_hours = random.randint(2, 8)

        # 15. 添加到效能记录
        equipment_efficiency_records.append({
            '日期': date,
            '设备ID': equipment_id,
            '理论产能(瓶)': theoretical_capacity,
            '实际产能(瓶)': actual_output,
            '产能利用率': round(performance, 3),
            '运行效率': round(availability, 3),
            '综合效率(OEE)': round(oee, 3),
            '故障次数': fault_count,
            '总故障时长(小时)': fault_hours,
            '备注': ''
        })

    # 16. 生成设备效能日报表
    equipment_efficiency_df = pd.DataFrame(equipment_efficiency_records)

    return equipment_status_df, equipment_efficiency_df


# ================== 使用示例 ==================
if __name__ == "__main__":
    # 1. 读取您的生产日报表和设备基础信息表
    # 替换为您的实际文件路径
    production_daily_df = pd.read_csv('production_daily.csv')
    equipment_df = pd.read_csv('equipment_base.csv')

    # 2. 生成报表
    equipment_status_df, equipment_efficiency_df = generate_equipment_reports(
        production_daily_df, equipment_df
    )

    # 3. 保存报表到CSV文件
    equipment_status_df.to_csv('equipment_status_daily.csv', index=False)
    equipment_efficiency_df.to_csv('equipment_efficiency_daily.csv', index=False)

    # 4. 输出报表概要（不包含验证）
    print("设备状态日报表已生成，保存为: equipment_status_daily.csv")
    print("设备效能日报表已生成，保存为: equipment_efficiency_daily.csv")

    # 5. 打印前5行供快速查看
    print("\n设备状态日报表预览:")
    print(equipment_status_df.head())

    print("\n设备效能日报表预览:")
    print(equipment_efficiency_df.head())