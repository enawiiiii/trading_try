#!/usr/bin/env python3
"""
Script to optimize Take Profit and Stop Loss parameters
by testing different combinations on historical data.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import replace
from trading_agent.config import load_config, AgentConfig
from trading_agent.historical_store import load_candles
from trading_agent.edge_filter import evaluate_edge_filter
from trading_agent.market import analyze_market
from trading_agent.strategies import select_signal
from trading_agent.risk import RiskContext, apply_risk_rules
from trading_agent.learning import load_learning_state
from trading_agent.trade_outcome import horizon_for_interval, long_outcome

def test_tp_sl_combinations(
    config_path: str = "config.json",
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    provider: str = "bybit",
    lookback: int = 80,
    max_events: int = 1000,
    stride: int = 6
):
    """Test different TP/SL combinations and find the best one."""
    
    config = load_config(config_path)
    candles = load_candles(config, provider, symbol, interval)
    horizon = horizon_for_interval(interval)
    
    if len(candles) < lookback + horizon + 1:
        print(f"❌ Not enough candles for {symbol} {interval}")
        return
    
    learning_state = load_learning_state(config)
    
    # Define TP/SL combinations to test
    tp_sl_configs = [
        {"tp": 1.0, "sl": 0.5},   # 2:1 ratio, tight
        {"tp": 1.5, "sl": 0.75},  # 2:1 ratio
        {"tp": 2.0, "sl": 1.0},   # 2:1 ratio (current)
        {"tp": 2.5, "sl": 1.0},   # 2.5:1 ratio
        {"tp": 3.0, "sl": 1.0},   # 3:1 ratio
        {"tp": 3.0, "sl": 1.5},   # 2:1 ratio, wider
        {"tp": 1.5, "sl": 0.5},   # 3:1 ratio, tight
        {"tp": 2.0, "sl": 0.8},   # 2.5:1 ratio
        {"tp": 1.2, "sl": 0.6},   # 2:1 ratio, very tight
        {"tp": 4.0, "sl": 1.0},   # 4:1 ratio
    ]
    
    print("=" * 80)
    print("اختبار تحسين معايير الخروج - TP/SL Optimization")
    print("=" * 80)
    print(f"\nالعملة: {symbol} | الفترة: {interval} | عدد الشموع: {len(candles)}")
    print(f"\nجاري اختبار {len(tp_sl_configs)} تركيبة مختلفة...\n")
    
    results = []
    
    for tp_sl in tp_sl_configs:
        tp = tp_sl["tp"]
        sl = tp_sl["sl"]
        
        # Create a new config with updated TP/SL using replace
        test_config = replace(config, take_profit_pct=tp, stop_loss_pct=sl)
        
        buy_trades = []
        total_events = 0
        
        max_start = len(candles) - horizon
        start = max(lookback, max_start - max_events * max(1, stride))
        
        for end in range(start, max_start, max(1, stride)):
            window = candles[end - lookback : end]
            future = candles[end : end + horizon]
            
            state = analyze_market(window)
            signal = select_signal(state, learning_state.strategy_weights, has_position=False)
            risk = apply_risk_rules(signal, state, test_config, RiskContext(has_position=False))
            
            edge = evaluate_edge_filter(
                symbol=symbol,
                interval=interval,
                state=state,
                signal=signal,
                risk=risk,
                config=test_config,
            )
            
            total_events += 1
            
            if risk.action.value == "BUY":
                outcome = long_outcome(
                    window[-1].close, 
                    future, 
                    test_config.take_profit_pct, 
                    test_config.stop_loss_pct
                )
                
                if "realized_pct" in outcome:
                    buy_trades.append({
                        "timestamp": candles[end - 1].close_time.isoformat(),
                        "entry_price": window[-1].close,
                        "realized_pct": outcome["realized_pct"],
                        "outcome": outcome.get("outcome", "unknown"),
                        "score": edge.score,
                    })
        
        # Calculate metrics
        if buy_trades:
            wins = [t for t in buy_trades if t["realized_pct"] > 0]
            losses = [t for t in buy_trades if t["realized_pct"] <= 0]
            
            win_rate = (len(wins) / len(buy_trades)) * 100
            avg_return = sum(t["realized_pct"] for t in buy_trades) / len(buy_trades)
            avg_win = sum(t["realized_pct"] for t in wins) / len(wins) if wins else 0
            avg_loss = sum(t["realized_pct"] for t in losses) / len(losses) if losses else 0
            
            # Calculate max drawdown
            cumulative = 0
            peak = 0
            max_drawdown = 0
            for trade in buy_trades:
                cumulative += trade["realized_pct"]
                if cumulative > peak:
                    peak = cumulative
                drawdown = (peak - cumulative)
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            # Calculate profit factor
            gross_profit = sum(t["realized_pct"] for t in wins)
            gross_loss = abs(sum(t["realized_pct"] for t in losses))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            results.append({
                "tp": tp,
                "sl": sl,
                "ratio": tp/sl,
                "total_trades": len(buy_trades),
                "win_rate": win_rate,
                "avg_return": avg_return,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "max_drawdown": max_drawdown,
                "profit_factor": profit_factor,
                "total_events": total_events,
            })
            
            status = "✅" if avg_return > 0 else "⚠️" if avg_return > -0.5 else "❌"
            print(f"{status} TP: {tp:.1f}% | SL: {sl:.1f}% | Ratio: {tp/sl:.1f}:1")
            print(f"   الصفقات: {len(buy_trades)} | الربح: {win_rate:.1f}% | العائد: {avg_return:.3f}%")
            print(f"   متوسط الربح: {avg_win:.3f}% | متوسط الخسارة: {avg_loss:.3f}%")
            print(f"   Max DD: {max_drawdown:.2f}% | Profit Factor: {profit_factor:.2f}")
            print()
        else:
            print(f"❌ TP: {tp:.1f}% | SL: {sl:.1f}% | لا توجد صفقات شراء")
            print()
    
    # Sort by average return
    if results:
        best_by_return = sorted(results, key=lambda x: x["avg_return"], reverse=True)[0]
        best_by_winrate = sorted(results, key=lambda x: x["win_rate"], reverse=True)[0]
        best_by_pf = sorted(results, key=lambda x: x["profit_factor"], reverse=True)[0]
        
        print("=" * 80)
        print("🏆 أفضل النتائج:")
        print("=" * 80)
        
        print(f"\n1️⃣ أعلى عائد متوسط:")
        print(f"   TP: {best_by_return['tp']:.1f}% | SL: {best_by_return['sl']:.1f}%")
        print(f"   العائد: {best_by_return['avg_return']:.3f}% | الربح: {best_by_return['win_rate']:.1f}%")
        print(f"   الصفقات: {best_by_return['total_trades']} | Profit Factor: {best_by_return['profit_factor']:.2f}")
        
        print(f"\n2️⃣ أعلى نسبة ربح:")
        print(f"   TP: {best_by_winrate['tp']:.1f}% | SL: {best_by_winrate['sl']:.1f}%")
        print(f"   الربح: {best_by_winrate['win_rate']:.1f}% | العائد: {best_by_winrate['avg_return']:.3f}%")
        print(f"   الصفقات: {best_by_winrate['total_trades']} | Profit Factor: {best_by_winrate['profit_factor']:.2f}")
        
        print(f"\n3️⃣ أفضل Profit Factor:")
        print(f"   TP: {best_by_pf['tp']:.1f}% | SL: {best_by_pf['sl']:.1f}%")
        print(f"   Profit Factor: {best_by_pf['profit_factor']:.2f} | العائد: {best_by_pf['avg_return']:.3f}%")
        print(f"   الربح: {best_by_pf['win_rate']:.1f}% | الصفقات: {best_by_pf['total_trades']}")
        
        # Recommendation
        print("\n" + "=" * 80)
        print("💡 التوصية:")
        print("=" * 80)
        
        if best_by_return['avg_return'] > 0 and best_by_return['profit_factor'] > 1.5:
            recommended = best_by_return
            reason = "أعلى عائد مع Profit Factor جيد"
        elif best_by_pf['profit_factor'] > 2.0:
            recommended = best_by_pf
            reason = "أفضل Profit Factor"
        else:
            recommended = best_by_winrate
            reason = "أعلى نسبة ربح للاستقرار"
        
        print(f"\n✅ الإعداد الموصى به:")
        print(f"   Take Profit: {recommended['tp']:.1f}%")
        print(f"   Stop Loss: {recommended['sl']:.1f}%")
        print(f"   السبب: {reason}")
        print(f"\n   الأداء المتوقع:")
        print(f"   - نسبة الربح: {recommended['win_rate']:.1f}%")
        print(f"   - متوسط العائد: {recommended['avg_return']:.3f}%")
        print(f"   - Profit Factor: {recommended['profit_factor']:.2f}")
        print(f"   - Max Drawdown: {recommended['max_drawdown']:.2f}%")
        
        # Save results
        output = {
            "symbol": symbol,
            "interval": interval,
            "tested_configurations": len(tp_sl_configs),
            "all_results": results,
            "best_by_return": best_by_return,
            "best_by_winrate": best_by_winrate,
            "best_by_profit_factor": best_by_pf,
            "recommended": recommended,
        }
        
        report_path = Path("data/tp_sl_optimization_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ تم حفظ التقرير الكامل في: {report_path}")
        
        return recommended
    
    return None

if __name__ == "__main__":
    test_tp_sl_combinations()
