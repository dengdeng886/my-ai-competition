import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional


class OptimizationEngine:
    def __init__(self, equipment_df: pd.DataFrame):
        """
        初始化优化引擎
        :param equipment_df: 设备基础信息表，包含 ['设备ID', '工序类型', '理论产能(瓶)', '区域ID', '可用率'] 等字段
        """
        self.equipment_df = equipment_df.copy()
        self.rules = self._initialize_rules()

    def _initialize_rules(self) -> Dict[str, Dict]:
        """初始化优化规则库"""
        return {
            'buffer_overflow_risk': {
                'condition': lambda data: data.get('safety_stock_ratio', 0) > 1.5,
                'actions': [
                    self._reduce_upstream_speed,
                    self._redirect_to_alternative_line,
                    self._increase_downstream_capacity
                ]
            },
            'buffer_empty_risk': {
                'condition': lambda data: data.get('safety_stock_ratio', 0) < 0.5,
                'actions': [
                    self._increase_upstream_speed,
                    self._prioritize_bottleneck_equipment
                ]
            },
            'bottleneck_detected': {
                'condition': lambda data: data.get('bottleneck_severity', 0) > 0.3,
                'actions': [
                    self._identify_bottleneck_equipment,
                    self._optimize_equipment_scheduling
                ]
            },
            'high_buffer_risk': {
                'condition': lambda data: data.get('buffer_risk_score', 0) > 0.8,
                'actions': [
                    self._balance_filling_packaging,
                    self._trigger_buffer_alert
                ]
            }
        }

    def analyze_situation(self, current_features: Dict[str, Any], predictions: Dict[str, float]) -> Dict[str, Any]:
        """
        分析当前生产状况并生成优化建议
        :param current_features: 当前特征字典（来自 build_features_from_raw）
        :param predictions: 模型预测结果，含 production_gap, filling_packaging_balance 等
        :return: 包含风险、建议和影响的报告
        """
        # 合并特征和预测，便于规则使用
        combined_data = {**current_features, **predictions}
        situation_report = {
            'risks': [],
            'recommendations': [],
            'predicted_impact': {}
        }

        triggered = False
        for risk_type, rule in self.rules.items():
            if rule['condition'](combined_data):
                situation_report['risks'].append(risk_type)
                for action in rule['actions']:
                    recommendation = action(combined_data, predictions)
                    if recommendation:
                        situation_report['recommendations'].append(recommendation)
                        triggered = True

        # 如果无风险，给出正向反馈
        if not triggered:
            situation_report['recommendations'].append({
                'action': '维持当前生产计划',
                'target': '全产线',
                'adjustment': '无需调整',
                'reason': '当前系统运行平稳，无显著风险',
                'expected_impact': '保持高效稳定生产'
            })

        return situation_report

    # ======================
    # 优化动作实现
    # ======================

    def _reduce_upstream_speed(self, features: Dict, predictions: Dict) -> Optional[Dict]:
        """降低上游（灌装）速度"""
        balance = predictions.get('filling_packaging_balance', 1.0)
        if balance > 1.2:  # 灌装远超包装
            return {
                'action': '降低灌装线速度',
                'target': '灌装设备组',
                'adjustment': '-10%',
                'reason': f'灌装/包装平衡比过高 ({balance:.2f})，缓冲区有溢出风险',
                'expected_impact': '降低缓冲区库存压力，提升系统稳定性'
            }
        return None

    def _increase_upstream_speed(self, features: Dict, predictions: Dict) -> Optional[Dict]:
        """提高上游速度以避免缓冲区空仓"""
        balance = predictions.get('filling_packaging_balance', 1.0)
        if balance < 0.8:  # 灌装不足
            return {
                'action': '提高灌装线速度',
                'target': '灌装设备组',
                'adjustment': '+10%',
                'reason': f'灌装/包装平衡比过低 ({balance:.2f})，缓冲区面临空仓风险',
                'expected_impact': '保障下游包装线连续供料'
            }
        return None

    def _redirect_to_alternative_line(self, features: Dict, predictions: Dict) -> Optional[Dict]:
        """重定向到备用包装线"""
        available_lines = self._find_available_packaging_lines()
        if available_lines:
            return {
                'action': '任务重分配',
                'target': f'包装线 {", ".join(available_lines[:2])}',
                'adjustment': '分流20%任务量',
                'reason': '利用空闲包装产能缓解瓶颈',
                'expected_impact': '提升整体包装吞吐量，降低 backlog'
            }
        return None

    def _increase_downstream_capacity(self, features: Dict, predictions: Dict) -> Optional[Dict]:
        """提升下游（包装）产能"""
        # 检查是否有可启用的备用设备
        standby_packers = self.equipment_df[
            (self.equipment_df['工序类型'] == '包装') &
            (self.equipment_df['可用率'] < 0.5)  # 假设可用率低表示备用
        ]['设备ID'].tolist()

        if standby_packers:
            return {
                'action': '启用备用包装设备',
                'target': f'{standby_packers[0]}',
                'adjustment': '启动并投入生产',
                'reason': '下游包装能力不足，需扩容',
                'expected_impact': '直接提升包装产能，改善灌包平衡'
            }
        return None

    def _identify_bottleneck_equipment(self, features: Dict, predictions: Dict) -> Optional[Dict]:
        """识别瓶颈设备"""
        low_eff_eq = self._find_low_efficiency_equipment()
        if low_eff_eq:
            return {
                'action': '设备性能优化',
                'target': f'设备 {", ".join(low_eff_eq)}',
                'adjustment': '优先维护/校准',
                'reason': '设备效率持续低于阈值，构成系统瓶颈',
                'expected_impact': '消除局部瓶颈，提升整体OEE'
            }
        return None

    def _optimize_equipment_scheduling(self, features: Dict, predictions: Dict) -> Optional[Dict]:
        """优化设备排程"""
        return {
            'action': '动态排程调整',
            'target': '全产线调度系统',
            'adjustment': '延长高效设备运行时间，缩短低效设备班次',
            'reason': '基于设备效率差异优化资源分配',
            'expected_impact': '提升单位时间产出，降低能耗'
        }

    def _prioritize_bottleneck_equipment(self, features: Dict, predictions: Dict) -> Optional[Dict]:
        """优先保障瓶颈设备物料/维护"""
        bottleneck_eq = self._find_low_efficiency_equipment()
        if bottleneck_eq:
            return {
                'action': '瓶颈设备优先保障',
                'target': f'{bottleneck_eq[0]}',
                'adjustment': '优先供料 + 实时监控',
                'reason': '防止瓶颈设备因缺料或故障停机',
                'expected_impact': '最大化瓶颈设备利用率'
            }
        return None

    def _balance_filling_packaging(self, features: Dict, predictions: Dict) -> Optional[Dict]:
        """主动平衡灌装与包装节奏"""
        balance = predictions.get('filling_packaging_balance', 1.0)
        if abs(balance - 1.0) > 0.3:
            direction = "降低灌装" if balance > 1.0 else "提升灌装"
            return {
                'action': '灌装-包装协同调速',
                'target': '灌装与包装联动控制',
                'adjustment': f'{direction}至平衡状态',
                'reason': f'缓冲区风险高 (buffer_risk={predictions.get("buffer_risk_score", 0):.2f})',
                'expected_impact': '实现动态平衡，降低缓冲区波动'
            }
        return None

    def _trigger_buffer_alert(self, features: Dict, predictions: Dict) -> Optional[Dict]:
        """触发缓冲区预警"""
        return {
            'action': '缓冲区风险预警',
            'target': '生产监控中心',
            'adjustment': '启动应急预案',
            'reason': f'缓冲区风险评分过高 ({predictions.get("buffer_risk_score", 0):.2f})',
            'expected_impact': '提前干预，避免停产'
        }

    # ======================
    # 辅助查询方法
    # ======================

    def _find_available_packaging_lines(self) -> List[str]:
        """查找当前可用的包装线（简化版）"""
        packaging_eq = self.equipment_df[self.equipment_df['工序类型'] == '包装']
        # 假设设备ID包含可用信息，或随机返回2个
        return packaging_eq['设备ID'].head(2).tolist()

    def _find_low_efficiency_equipment(self, threshold: float = 0.7) -> List[str]:
        """查找效率低于阈值的设备（模拟）"""
        # 实际中应从实时效率表查询，此处模拟返回固定设备
        return ['FILLER_05', 'PACKAGER_A_03']

    # ======================
    # 影响模拟（用于前端展示）
    # ======================

    def simulate_optimization_impact(self, recommendation: Dict, current_state: Dict) -> Dict[str, Any]:
        """模拟单条建议的优化影响"""
        impact = {
            'recommendation': recommendation,
            'before': current_state.copy(),
            'after': {},
            'improvements': {}
        }

        after = current_state.copy()
        action = recommendation['action']

        # 模拟状态变化
        if '降低灌装线速度' in action:
            after['filling_packaging_balance'] = max(0.5, after.get('filling_packaging_balance', 1.0) - 0.3)
            after['buffer_risk_score'] = max(0.0, after.get('buffer_risk_score', 0.9) - 0.25)
        elif '提高灌装线速度' in action:
            after['filling_packaging_balance'] = min(1.5, after.get('filling_packaging_balance', 1.0) + 0.3)
            after['buffer_risk_score'] = max(0.0, after.get('buffer_risk_score', 0.9) - 0.2)
        elif '任务重分配' in action or '启用备用' in action:
            after['bottleneck_severity'] = max(0.0, after.get('bottleneck_severity', 0.5) - 0.15)
            after['production_gap'] = min(0.0, after.get('production_gap', -0.02) + 0.01)
        elif '灌装-包装协同调速' in action:
            after['filling_packaging_balance'] = 1.0
            after['buffer_risk_score'] = 0.5

        impact['after'] = after

        # 计算改进量
        for key in ['buffer_risk_score', 'bottleneck_severity', 'production_gap', 'filling_packaging_balance']:
            if key in current_state and key in after:
                delta = after[key] - current_state[key]
                if abs(delta) > 1e-4:
                    impact['improvements'][key] = delta

        return impact