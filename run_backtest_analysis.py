#!/usr/bin/env python3
"""
Script to run a full backtest simulation on historical data
and generate a comprehensive report for demo account readiness.
"""

import json
from pathlib import Path
from trading_agent.config import load_config
from trading_agent.edge_backtest import backtest_edge_filter
from trading_agent.historical_store import load_candles
from trading_agent.backtest import backtest_probabilities

def run_full_backtest_report(config_path: str = "config.json"):
    """Run comprehensive backtests and generate a summary report."""
    
    config = load_config(config_path)
    
    print("=" * 60)
    print("تقرير جاهزية الحساب التجريبي - Backtest Readiness Report")
    print("=" * 60)
    
    # 1. Run Edge Filter Backtest on multiple symbols
    print("\n[1] جاري اختبار فلتر الحافة (Edge Filter) على بيانات تاريخية...")
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    intervals = ["1h", "4h"]
    
    edge_report = backtest_edge_filter(
        config=config,
        provider="bybit",
        symbols=symbols,
        intervals=intervals,
        lookback=80,
        max_events_per_model=500,
        stride=6
    )
    
    summary = edge_report.get("summary", {})
    
    print(f"\n✓ تم تحليل {summary.get('events', 0)} حدث عبر {len(symbols)} عملات")
    print(f"✓ إشارات الشراء المكتشفة: {summary.get('buy', 0)}")
    print(f"✓ متوسط النقاط (Score): {summary.get('avg_score', 0):.2f}")
    print(f"✓ أحداث الشراء الكلية: {summary.get('buy_events', 0)}")
    
    if summary.get('buy_events', 0) > 0:
        avg_realized = summary.get('all_buy_avg_realized_pct', 0)
        win_rate = summary.get('all_buy_win_rate_pct', 0)
        print(f"✓ متوسط العائد لكل صفقة: {avg_realized:.4f}%")
        print(f"✓ نسبة الصفقات الرابحة: {win_rate:.2f}%")
        
        # Check allowed buys specifically
        allowed_events = summary.get('allowed_buy_events', 0)
        if allowed_events > 0:
            allowed_avg = summary.get('allowed_buy_avg_realized_pct', 0)
            allowed_win = summary.get('allowed_buy_win_rate_pct', 0)
            print(f"\n[مهم] الصفقات المسموحة فقط (بعد الفلتر):")
            print(f"  - العدد: {allowed_events}")
            print(f"  - متوسط العائد: {allowed_avg:.4f}%")
            print(f"  - نسبة الربح: {allowed_win:.2f}%")
    
    # 2. Run Probability Backtest on BTC
    print("\n[2] جاري اختبار دقة التنبؤات على BTCUSDT...")
    btc_candles = load_candles(config, "bybit", "BTCUSDT", "1h")
    
    if len(btc_candles) > 100:
        prob_report = backtest_probabilities(btc_candles, lookback=80, horizon=6, move_threshold_pct=0.35)
        
        accuracy = prob_report.get('accuracy', 0)
        high_conf_acc = prob_report.get('high_confidence_accuracy', 0)
        high_conf_samples = prob_report.get('high_confidence_samples', 0)
        
        print(f"✓ عدد العينات المختبرة: {prob_report.get('samples', 0)}")
        print(f"✓ الدقة العامة: {accuracy:.2f}%")
        print(f"✓ دقة الثقة العالية ({high_conf_samples} عينة): {high_conf_acc:.2f}%")
        
        # Show by scenario
        by_scenario = prob_report.get('by_scenario', {})
        if by_scenario:
            print("\nالدقة حسب السيناريو:")
            for scenario, stats in by_scenario.items():
                print(f"  - {scenario}: {stats.get('accuracy', 0):.2f}% ({stats.get('samples', 0)} عينة)")
    
    # 3. Generate Readiness Assessment
    print("\n" + "=" * 60)
    print("تقييم الجاهزية للحساب التجريبي")
    print("=" * 60)
    
    readiness_score = 0
    recommendations = []
    
    # Criteria 1: Enough data
    if summary.get('events', 0) >= 1000:
        readiness_score += 20
        print("✓ [PASS] حجم بيانات كافٍ للاختبار")
    else:
        recommendations.append("⚠ زيادة حجم البيانات التاريخية المختبرة")
        print("⚠ [WARN] حجم البيانات قد يكون صغيراً")
    
    # Criteria 2: Win rate
    if summary.get('all_buy_win_rate_pct', 0) >= 45:
        readiness_score += 25
        print("✓ [PASS] نسبة ربح مقبولة (>45%)")
    elif summary.get('all_buy_win_rate_pct', 0) >= 35:
        readiness_score += 15
        recommendations.append("⚠ تحسين استراتيجية الدخول لرفع نسبة الربح")
        print("⚠ [NEEDS IMPROVEMENT] نسبة الربح تحتاج تحسين (35-45%)")
    else:
        recommendations.append("❌ إعادة تدريب النموذج أو تعديل المعايير")
        print("❌ [FAIL] نسبة الربح منخفضة جداً (<35%)")
    
    # Criteria 3: Positive EV
    if summary.get('all_buy_avg_realized_pct', 0) > 0:
        readiness_score += 25
        print("✓ [PASS] عائد متوسط موجب")
    else:
        recommendations.append("⚠ تعديل معايير الخروج (Take Profit / Stop Loss)")
        print("⚠ [WARN] العائد المتوسط سلبي أو صفري")
    
    # Criteria 4: Allowed buys performance
    if summary.get('allowed_buy_events', 0) > 0:
        if summary.get('allowed_buy_avg_realized_pct', 0) > 0:
            readiness_score += 20
            print("✓ [PASS] الفلتر ينتقي صفقات مربحة")
        else:
            recommendations.append("⚠ ضبط عتبة فلتر الحافة (Edge Filter)")
            print("⚠ [WARN] الصفقات المسموحة غير مربحة")
    else:
        recommendations.append("⚠ تخفيف قيود الفلتر لرؤية نتائج أكثر")
        print("⚠ [INFO] لا توجد صفقات مسموحة كافية للتقييم")
    
    # Criteria 5: ML Accuracy
    if 'high_conf_acc' in locals() and high_conf_acc >= 40:
        readiness_score += 10
        print("✓ [PASS] دقة النموذج مقبولة")
    else:
        recommendations.append("⚠ مزيد من التدريب للنموذج")
        print("⚠ [INFO] دقة النموذج تحتاج تحسين")
    
    print(f"\n{'=' * 60}")
    print(f"النتيجة الإجمالية: {readiness_score}/100")
    
    if readiness_score >= 70:
        print("\n🎉 التوصية: جاهز للحساب التجريبي!")
        print("   ابدأ بمبلغ صغير وراقب الأداء عن كثب.")
    elif readiness_score >= 50:
        print("\n⚠ التوصية: يحتاج تحسينات قبل الحساب التجريبي")
        print("   نفذ التوصيات أدناه ثم أعد الاختبار.")
    else:
        print("\n❌ التوصية: غير جاهز بعد")
        print("   تحتاج إلى تحسينات جوهرية في الاستراتيجية أو النموذج.")
    
    if recommendations:
        print(f"\n{'=' * 60}")
        print("التوصيات للتحسين:")
        for rec in recommendations:
            print(f"  {rec}")
    
    print(f"\n{'=' * 60}")
    print("ملاحظة هامة:")
    print("  - النتائج السابقة لا تضمن الأرباح المستقبلية")
    print("  - ابدأ دائماً بأموال يمكنك تحمل خسارتها")
    print("  - استخدم وقف الخسارة دائماً")
    print(f"{'=' * 60}\n")
    
    # Save report
    report_path = Path("data/backtest_readiness_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    full_report = {
        "edge_filter_summary": summary,
        "probability_report": prob_report if 'prob_report' in locals() else None,
        "readiness_score": readiness_score,
        "recommendations": recommendations
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    
    print(f"✓ تم حفظ التقرير الكامل في: {report_path}\n")
    
    return readiness_score >= 70

if __name__ == "__main__":
    run_full_backtest_report()
