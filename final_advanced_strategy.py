#!/usr/bin/env python3
"""
استراتيجية تداول متقدمة جداً مع فلاتر متعددة
تستخدم: RSI, MACD, Bollinger Bands, EMA, Volume, ATR, Stochastic
مع نظام نقاط متطور للدخول
"""

import pandas as pd
import numpy as np
import ccxt
import talib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("استراتيجية التداول المتقدمة - Multi-Indicator Strategy")
print("="*80)

def fetch_data(symbol, timeframe='4h', limit=5000):
    """جلب البيانات"""
    exchange = ccxt.bybit({
        'enableRateLimit': True,
        'options': {'defaultType': 'linear'}
    })
    
    print(f"\n[*] جاري جلب {limit} شمعة لـ {symbol} على timeframe {timeframe}...")
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    return df

def add_all_indicators(df):
    """إضافة جميع المؤشرات الفنية"""
    # RSI
    df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
    df['rsi_7'] = talib.RSI(df['close'], timeperiod=7)
    
    # MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
    
    # Bollinger Bands
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # EMA
    df['ema_9'] = talib.EMA(df['close'], timeperiod=9)
    df['ema_21'] = talib.EMA(df['close'], timeperiod=21)
    df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
    df['ema_200'] = talib.EMA(df['close'], timeperiod=200)
    
    # ATR
    df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    
    # Volume
    df['volume_sma_20'] = talib.SMA(df['volume'], timeperiod=20)
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    # Stochastic
    df['slowk'], df['slowd'] = talib.STOCH(df['high'], df['low'], df['close'])
    
    # ADX
    df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    
    # CCI
    df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)
    
    # Williams %R
    df['willr'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
    
    # MFI
    df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
    
    # OBV
    df['obv'] = talib.OBV(df['close'], df['volume'])
    df['obv_sma'] = talib.SMA(df['obv'], timeperiod=20)
    
    # Price position
    df['price_vs_ema200'] = (df['close'] - df['ema_200']) / df['ema_200']
    
    return df

def calculate_entry_score(row, prev_row):
    """حساب نقطة الدخول بناءً على مؤشرات متعددة"""
    if pd.isna(row['rsi_14']) or pd.isna(row['macd']) or pd.isna(row['adx']):
        return 0
    
    score = 0
    max_score = 0
    
    # 1. الاتجاه العام (EMA) - 15 نقطة
    max_score += 15
    if row['ema_9'] > row['ema_21'] > row['ema_50']:
        score += 15  # اتجاه صاعد قوي
    elif row['ema_9'] > row['ema_21']:
        score += 10  # اتجاه صاعد متوسط
    elif row['ema_9'] > row['ema_50']:
        score += 5   # اتجاه صاعد ضعيف
    
    # 2. السعر فوق EMA 200 - 10 نقاط
    max_score += 10
    if row['close'] > row['ema_200']:
        score += 10
    
    # 3. MACD - 15 نقطة
    max_score += 15
    if row['macd'] > row['macd_signal']:
        score += 8
        if row['macd_hist'] > 0 and row['macd_hist'] > (prev_row['macd_hist'] if not pd.isna(prev_row['macd_hist']) else 0):
            score += 7  # زخم إيجابي متزايد
    
    # 4. RSI - 15 نقطة
    max_score += 15
    if 40 <= row['rsi_14'] <= 60:
        score += 10  # منطقة مثالية
    elif 30 <= row['rsi_14'] < 40:
        score += 12  # تشبع بيعي خفيف
    elif row['rsi_14'] < 30:
        score += 15  # تشبع بيعي قوي
    elif 60 < row['rsi_14'] <= 70:
        score += 5
    
    # 5. Bollinger Bands - 10 نقاط
    max_score += 10
    if row['bb_position'] < 0.3:
        score += 10  # السعر قرب الحد السفلي
    elif row['bb_position'] < 0.5:
        score += 6
    
    # 6. Volume - 10 نقاط
    max_score += 10
    if row['volume_ratio'] > 1.5:
        score += 10
    elif row['volume_ratio'] > 1.2:
        score += 6
    elif row['volume_ratio'] > 1.0:
        score += 3
    
    # 7. Stochastic - 10 نقاط
    max_score += 10
    if row['slowk'] < 20 and row['slowd'] < 20:
        score += 10  # تشبع بيعي
    elif row['slowk'] < 30 and row['slowd'] < 30:
        score += 7
    elif row['slowk'] > row['slowd'] and row['slowk'] < 50:
        score += 5
    
    # 8. ADX - 5 نقاط
    max_score += 5
    if row['adx'] > 25:
        score += 5  # اتجاه قوي
    elif row['adx'] > 20:
        score += 3
    
    # 9. CCI - 5 نقاط
    max_score += 5
    if -100 <= row['cci'] <= 0:
        score += 5
    elif row['cci'] < -100:
        score += 3
    
    # 10. Williams %R - 5 نقاط
    max_score += 5
    if row['willr'] < -80:
        score += 5
    elif row['willr'] < -50:
        score += 2
    
    # النسبة المئوية
    return (score / max_score) * 100 if max_score > 0 else 0

def advanced_backtest(df, tp_pct, sl_pct, min_score=70, initial_capital=10000):
    """اختبار الاستراتيجية المتقدمة"""
    capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    in_position = False
    
    for i in range(250, len(df) - 10):  # نحتاج بيانات سابقة للمؤشرات
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        if in_position:
            high = row['high']
            low = row['low']
            
            # Take Profit
            if high >= entry_price * (1 + tp_pct):
                exit_price = entry_price * (1 + tp_pct)
                pnl = position * (exit_price - entry_price)
                capital += pnl
                trades.append({'entry': entry_price, 'exit': exit_price, 'pnl_pct': tp_pct, 'result': 'WIN'})
                in_position = False
                position = 0
            # Stop Loss
            elif low <= entry_price * (1 - sl_pct):
                exit_price = entry_price * (1 - sl_pct)
                pnl = position * (exit_price - entry_price)
                capital += pnl
                trades.append({'entry': entry_price, 'exit': exit_price, 'pnl_pct': -sl_pct, 'result': 'LOSS'})
                in_position = False
                position = 0
        
        if not in_position:
            score = calculate_entry_score(row, prev_row)
            if score >= min_score:
                risk_amount = capital * 0.02
                stop_loss_distance = row['close'] * sl_pct
                if stop_loss_distance > 0:
                    position_size = risk_amount / stop_loss_distance
                    position = position_size
                    entry_price = row['close']
                    in_position = True
    
    # إغلاق المركز الأخير
    if in_position:
        final_price = df.iloc[-1]['close']
        pnl = position * (final_price - entry_price)
        capital += pnl
        pnl_pct = (final_price - entry_price) / entry_price
        trades.append({'entry': entry_price, 'exit': final_price, 'pnl_pct': pnl_pct, 'result': 'CLOSED'})
    
    return capital, trades

# === التنفيذ الرئيسي ===
symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
timeframes = ['4h', '1d']
results_summary = {}

for symbol in symbols:
    print(f"\n{'='*80}")
    print(f"معالجة {symbol}")
    print('='*80)
    
    best_result = None
    best_return = -float('inf')
    
    for tf in timeframes:
        try:
            df = fetch_data(symbol, timeframe=tf, limit=3000)
            df = add_all_indicators(df)
            df.dropna(inplace=True)
            
            print(f"\n[*] تم تجهيز {len(df)} شمعة بـ {len(df.columns)} مؤشر")
            
            # اختبار تركيبات مختلفة
            test_configs = [
                {'tp': 0.02, 'sl': 0.01, 'min_score': 65},
                {'tp': 0.03, 'sl': 0.015, 'min_score': 65},
                {'tp': 0.04, 'sl': 0.02, 'min_score': 65},
                {'tp': 0.02, 'sl': 0.01, 'min_score': 70},
                {'tp': 0.03, 'sl': 0.015, 'min_score': 70},
                {'tp': 0.04, 'sl': 0.02, 'min_score': 70},
                {'tp': 0.05, 'sl': 0.02, 'min_score': 70},
                {'tp': 0.03, 'sl': 0.01, 'min_score': 75},
                {'tp': 0.04, 'sl': 0.015, 'min_score': 75},
            ]
            
            for config in test_configs:
                final_capital, trades = advanced_backtest(
                    df, 
                    tp_pct=config['tp'], 
                    sl_pct=config['sl'], 
                    min_score=config['min_score']
                )
                
                total_trades = len(trades)
                if total_trades == 0:
                    continue
                
                winning_trades = [t for t in trades if t['result'] == 'WIN']
                losing_trades = [t for t in trades if t['result'] == 'LOSS']
                
                win_rate = len(winning_trades) / total_trades * 100
                total_return = (final_capital - 10000) / 10000 * 100
                
                gross_profit = sum([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
                gross_loss = abs(sum([t['pnl_pct'] for t in losing_trades])) if losing_trades else 0
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                
                if total_return > best_return:
                    best_return = total_return
                    best_result = {
                        'timeframe': tf,
                        'tp': config['tp'],
                        'sl': config['sl'],
                        'min_score': config['min_score'],
                        'total_trades': total_trades,
                        'win_rate': win_rate,
                        'total_return': total_return,
                        'final_capital': final_capital,
                        'profit_factor': profit_factor
                    }
        except Exception as e:
            print(f"خطأ في {symbol} على {tf}: {e}")
            continue
    
    if best_result:
        results_summary[symbol] = best_result
        print(f"\n{'='*80}")
        print(f"🏆 أفضل نتيجة لـ {symbol}:")
        print('='*80)
        print(f"Timeframe: {best_result['timeframe']}")
        print(f"TP: {best_result['tp']*100:.1f}%, SL: {best_result['sl']*100:.1f}%")
        print(f"الحد الأدنى للنقاط: {best_result['min_score']}")
        print(f"عدد الصفقات: {best_result['total_trades']}")
        print(f"نسبة الربح: {best_result['win_rate']:.1f}%")
        print(f"العائد الكلي: {best_result['total_return']:+.2f}%")
        print(f"رأس المال النهائي: ${best_result['final_capital']:.2f}")
        print(f"Profit Factor: {best_result['profit_factor']:.2f}")

# الملخص العام
print(f"\n{'='*80}")
print("الملخص العام النهائي")
print('='*80)
print(f"{'العملة':<20} {'TF':<6} {'صفقات':<8} {'Win%':<8} {'العائد%':<12} {'PF':<8}")
print('-'*80)

total_return_sum = 0
positive_count = 0

for symbol, res in results_summary.items():
    print(f"{symbol:<20} {res['timeframe']:<6} {res['total_trades']:<8} {res['win_rate']:.1f}%     {res['total_return']:+.2f}%      {res['profit_factor']:.2f}")
    total_return_sum += res['total_return']
    if res['total_return'] > 0:
        positive_count += 1

avg_return = total_return_sum / len(results_summary) if results_summary else 0

print(f"\n{'='*80}")
print("التقييم النهائي")
print('='*80)

if positive_count >= 2 and avg_return > 0:
    print("✅ النتائج إيجابية! يمكن التفكير في الاختبار على الحساب التجريبي.")
elif positive_count >= 1 and avg_return > -5:
    print("⚠️ النتائج مختلطة. تحتاج لمزيد من التحسين قبل الحساب التجريبي.")
else:
    print("❌ النتائج سلبية بشكل عام. يجب إعادة تصميم الاستراتيجية.")

print("\nملاحظات:")
print("- Profit Factor > 1.5 يعتبر جيد للتداول الحقيقي")
print("- Win Rate > 45% مع PF > 1.2 هو هدف معقول")
print("- العائد الإيجابي عبر عملات متعددة يدل على قوة الاستراتيجية")
print("- اختبر على فترات زمنية أطول (سنة+) للحصول على نتائج أكثر دقة")

print("\n" + "="*80)
print("تم الانتهاء من التحليل الشامل")
print("="*80)
