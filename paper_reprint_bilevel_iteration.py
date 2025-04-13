# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 09:45:56 2025

@author: lenovo
"""

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.core import Suffix
import numpy as np
import gurobipy

# ---------------------------- 数据输入 ----------------------------
generators = {
    'G1': {'node': 1, 'type': 'coal', 'c': 0.9, 'marginal_cost': 40, 
           'P_max': 100, 'P_min': 20, 'P0': 90/0.9},  # 无碳限制下出力100MW
    'G2': {'node': 2, 'type': 'hydro', 'c': 0.0, 'marginal_cost': 20, 
           'P_max': 170, 'P_min': 0, 'P0': 170},
    'G3': {'node': 3, 'type': 'gas', 'c': 0.3, 'marginal_cost': 50,
           'P_max': 520, 'P_min': 100, 'P0': 303.3},
    'G4': {'node': 4, 'type': 'gas', 'c': 0.4, 'marginal_cost': 65,
           'P_max': 200, 'P_min': 50, 'P0': 200},
    'G5': {'node': 5, 'type': 'coal', 'c': 0.8, 'marginal_cost': 35,
           'P_max': 600, 'P_min': 100, 'P0': 127.5}
}

consumers = {
    'L1': {'node': 3, 'marginal_utility': 93.5, 'P_max': 250},
    'L2': {'node': 4, 'marginal_utility': 88.0, 'P_max': 300},
    'L3': {'node': 5, 'marginal_utility': 82.5, 'P_max': 350}
}

lines = {
    (1,2): {'B': 1/0.03, 'P_max': 800},
    (1,4): {'B': 1/0.06, 'P_max': 100},
    (2,3): {'B': 1/0.07, 'P_max': 800},
    (3,4): {'B': 1/0.08, 'P_max': 800},
    (4,5): {'B': 1/0.02, 'P_max': 240},
    (3,5): {'B': 1/0.05, 'P_max': 800}
}

nodes = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
ref_node = 1  # 参考节点

# 计算初始碳排放权(E0)
total_emission_unlimited = sum(g['c']*g['P0'] for g in generators.values())
for g in generators.values():
    g['E0'] = (g['c']*g['P0']/total_emission_unlimited) * total_emission_unlimited

# ---------------------------- 市场出清模型 ----------------------------
def market_clearing(a, b, M, generators, consumers, lines, nodes):
    model = pyo.ConcreteModel()
    model.dual = Suffix(direction=Suffix.IMPORT) #显示调用对偶变量
    
    # 变量定义
    model.P_G = pyo.Var(generators.keys(), within=pyo.NonNegativeReals)
    model.P_D = pyo.Var(consumers.keys(), within=pyo.NonNegativeReals)
    model.delta = pyo.Var(nodes.keys(), within=pyo.Reals)
    
    # 目标函数: 最大化社会福利
    model.obj = pyo.Objective(
        expr=sum(b[j] * model.P_D[j] for j in consumers) - 
             sum(a[i] * model.P_G[i] for i in generators),
        sense=pyo.maximize
    )
    
    # 节点功率平衡约束
    def power_balance(model, n):
        return (
            sum(model.P_D[j] for j,c in consumers.items() if c['node']==n) -
            sum(model.P_G[i] for i,g in generators.items() if g['node']==n) +
            sum(lines[(n,m)]['B']*(model.delta[n]-model.delta[m]) 
                for m in nodes if (n,m) in lines)
            == 0
        )
    model.balance = pyo.Constraint(nodes.keys(), rule=power_balance)
    
    # 系统碳排放约束
    model.carbon = pyo.Constraint(
        expr=sum(g['c']*model.P_G[i] for i,g in generators.items()) <= M
    )
    
    # 机组出力约束
    def gen_limits(model, i):
        g = generators[i]
        return (g['P_min'], model.P_G[i], g['P_max'])
    model.gen_cons = pyo.Constraint(generators.keys(), rule=gen_limits)
    
    # 用户需求约束
    def demand_limits(model, j):
        return (0, model.P_D[j], consumers[j]['P_max'])
    model.demand_cons = pyo.Constraint(consumers.keys(), rule=demand_limits)
    
    # 线路潮流约束
    def line_flow(model, n, m):
        if (n,m) in lines:
            line = lines[(n,m)]
            return (-line['P_max'], line['B']*(model.delta[n]-model.delta[m]), line['P_max'])
        else:
            return pyo.Constraint.Skip
    model.line_cons = pyo.Constraint(nodes.keys(), nodes.keys(), rule=line_flow)
    
    # 参考节点相角
    model.ref = pyo.Constraint(expr=model.delta[ref_node] == 0)
    
    # 选择求解器
    solver = pyo.SolverFactory('gurobi')
    # 配置Gurobi参数
    solver.options['NonConvex'] = 2  # 允许非凸模型
    # 求解
    results = solver.solve(model)
    
    # 提取结果
    P_G = {i: pyo.value(model.P_G[i]) for i in generators}
    P_D = {j: pyo.value(model.P_D[j]) for j in consumers}
    lambda_n = {n: model.dual[model.balance[n]] for n in nodes}
    omega = model.dual[model.carbon] if model.carbon.active else 0
    
    return P_G, P_D, lambda_n, omega

# ---------------------------- 发电商优化模型 ----------------------------
def generator_optimize(gen_id, a_current, b_current, M, generators, consumers):
    gen = generators[gen_id]
    model = pyo.ConcreteModel()
    
    # 决策变量: 该发电商机组报价
    model.a = pyo.Var(within=pyo.NonNegativeReals)
    
    # 报价约束 (边际成本的±20%)
    model.a_lb = pyo.Constraint(expr=model.a >= 0.8 * gen['marginal_cost'])
    model.a_ub = pyo.Constraint(expr=model.a <= 1.2 * gen['marginal_cost'])
    
    # 固定其他参与者报价
    a_new = a_current.copy()
    
    # 定义目标函数
    def objective(model):
        a_new[gen_id] = model.a
        P_G, P_D, lambda_n, omega = market_clearing(a_new, b_current, M, generators, consumers, lines, nodes)
        
        # 计算该发电商收益
        revenue = lambda_n[gen['node']] * P_G[gen_id]  # 电能收益
        carbon_profit = omega * (gen['E0'] - gen['c'] * P_G[gen_id])  # 碳交易收益
        cost = gen['marginal_cost'] * P_G[gen_id]     # 发电成本
        total_profit = revenue + carbon_profit - cost
        
        return total_profit
    
    model.obj = pyo.Objective(rule=objective, sense=pyo.maximize)
    
    # 求解
    solver = pyo.SolverFactory('gurobi')
    solver.solve(model)
    
    return pyo.value(model.a)

# ---------------------------- 电力用户优化模型 ----------------------------
def consumer_optimize(cons_id, a_current, b_current, M, generators, consumers):
    cons = consumers[cons_id]
    model = pyo.ConcreteModel()
    
    # 决策变量: 该用户报价
    model.b = pyo.Var(within=pyo.NonNegativeReals)
    
    # 报价约束 (边际效用的±20%)
    model.b_lb = pyo.Constraint(expr=model.b >= 0.8 * cons['marginal_utility'])
    model.b_ub = pyo.Constraint(expr=model.b <= 1.2 * cons['marginal_utility'])
    
    # 固定其他参与者报价
    b_new = b_current.copy()
    
    # 定义目标函数
    def objective(model):
        b_new[cons_id] = model.b
        P_G, P_D, lambda_n, omega = market_clearing(a_current, b_new, M, generators, consumers, lines, nodes)
        
        # 计算该用户效用
        utility = cons['marginal_utility'] * P_D[cons_id]  # 用电效用
        cost = lambda_n[cons['node']] * P_D[cons_id]       # 购电成本
        total_utility = utility - cost
        
        return total_utility
    
    model.obj = pyo.Objective(rule=objective, sense=pyo.maximize)
    
    # 求解
    solver = pyo.SolverFactory('gurobi')
    solver.solve(model)
    
    return pyo.value(model.b)

# ---------------------------- 对角化迭代算法 ----------------------------
def diagonalized_iteration(M, generators, consumers, max_iter=20, tol=1e-3):
    # 初始化报价
    a = {i: g['marginal_cost'] for i,g in generators.items()}
    b = {j: c['marginal_utility'] for j,c in consumers.items()}
    
    for iter in range(max_iter):
        a_old = a.copy()
        b_old = b.copy()
        
        # 发电商优化阶段
        for gen_id in generators:
            # 固定其他报价，优化当前发电商报价
            best_a = a[gen_id]
            best_profit = -float('inf')
            
            # 定义报价搜索范围（示例：分10个点）
            a_min = 0.8 * generators[gen_id]['marginal_cost']
            a_max = 1.2 * generators[gen_id]['marginal_cost']
            a_candidates = np.linspace(a_min, a_max, 10)
            
            for a_candidate in a_candidates:
                # 临时更新报价
                a_temp = a.copy()
                a_temp[gen_id] = a_candidate
                
                # 求解市场出清
                try:
                    P_G, P_D, lambda_n, omega = market_clearing(
                        a_temp, b, M, generators, consumers, lines, nodes
                    )
                except:
                    continue  # 忽略求解失败的情况
                
                # 计算当前发电商收益
                revenue = lambda_n[generators[gen_id]['node']] * P_G[gen_id]
                carbon_profit = omega * (generators[gen_id]['E0'] - generators[gen_id]['c'] * P_G[gen_id])
                cost = generators[gen_id]['marginal_cost'] * P_G[gen_id]
                profit = revenue + carbon_profit - cost
                
                # 更新最优报价
                if profit > best_profit:
                    best_profit = profit
                    best_a = a_candidate
            
            a[gen_id] = best_a  # 更新报价
            
            #a[gen_id] = generator_optimize(gen_id, a, b, M, generators, consumers)
        
        # 用户优化阶段
        for cons_id in consumers:
            # 固定其他报价，优化当前电力用户报价
            best_b = b[cons_id]
            best_utility = -float('inf')
            
            # 定义报价搜索范围（示例：分10个点）
            b_min = 0.8 * consumers[cons_id]['marginal_utility']
            b_max = 1.2 * consumers[cons_id]['marginal_utility']
            b_candidates = np.linspace(b_min, b_max, 10)
            
            for b_candidate in b_candidates:
                # 临时更新报价
                b_temp = b.copy()
                b_temp[cons_id] = b_candidate
                
                # 求解市场出清
                try:
                    P_G, P_D, lambda_n, omega = market_clearing(
                        a, b_temp, M, generators, consumers, lines, nodes
                    )
                except:
                    continue  # 忽略求解失败的情况
                
                # 计算该用户效用
                utility = consumers[cons_id]['marginal_utility'] * P_D[cons_id]  # 用电效用
                cost = lambda_n[consumers[cons_id]['node']] * P_D[cons_id]       # 购电成本
                total_utility = utility - cost
                # 更新最优报价
                if total_utility > best_utility:
                    best_utility = total_utility
                    best_b = b_candidate
            
            b[cons_id] = best_b  # 更新报价
            
            #b[cons_id] = consumer_optimize(cons_id, a, b, M, generators, consumers)
        
        # 检查收敛
        a_diff = max(abs(a[i]-a_old[i]) for i in generators)
        b_diff = max(abs(b[j]-b_old[j]) for j in consumers)
        
        print(f"Iter {iter+1}: a_diff={a_diff:.4f}, b_diff={b_diff:.4f}")
        if a_diff < tol and b_diff < tol:
            break
    
    # 最终市场出清
    P_G, P_D, lambda_n, omega = market_clearing(a, b, M, generators, consumers, lines, nodes)
    return a, b, P_G, P_D, lambda_n, omega

# ---------------------------- 运行分析 ----------------------------
if __name__ == "__main__":
    # 计算无碳限制总排放
    total_emission_unlimited = sum(g['c']*g['P0'] for g in generators.values())
    
    # 定义三个碳排放限额场景
    scenarios = {
        '85%': 0.85 * total_emission_unlimited,
        '65%': 0.65 * total_emission_unlimited,
        '45%': 0.45 * total_emission_unlimited
    }
    
    # 运行各场景
    for scenario, M in scenarios.items():
        print(f"\n========== Scenario: {scenario} ==========")
        a, b, P_G, P_D, lambda_n, omega = diagonalized_iteration(M, generators, consumers)
        
        # 打印结果
        print("\nGenerator Bids:")
        for i in generators:
            print(f"{i}: {a[i]:.2f} $/MWh (原始成本: {generators[i]['marginal_cost']} $/MWh)")
        
        print("\nConsumer Bids:")
        for j in consumers:
            print(f"{j}: {b[j]:.2f} $/MWh (原始效用: {consumers[j]['marginal_utility']} $/MWh)")
        
        print("\nGeneration Outputs:")
        for i in generators:
            print(f"{i}: {P_G[i]:.1f} MW")
        
        print("\nLoad Serving:")
        for j in consumers:
            print(f"{j}: {P_D[j]:.1f}/{consumers[j]['P_max']} MW")
        
        print(f"\nCarbon Price: {omega:.2f} $/tCO2")
        print(f"System LMPs: {lambda_n}")
        print("="*50)