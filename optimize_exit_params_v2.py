#!/usr/bin/env python3
"""
تحسين معايير الخروج - بحث شامل عن أفضل TP/SL
مع تحليل مفصل للنتائج
"""

import pandas as pd
import numpy as np
import ccxt
import talib
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("نظام تحسين معايير الخروج - Advanced Exit Optimization")
print("="*70)

def fetch_ohlcv(symbol, timeframe='1h', limit=5000):
    """جلب بيانات OHLCV"""
    exchange = ccxt.bybit({
        'enableRateLimit': True,
        'options': {'defaultType': 'linear'}
    })
    
    print(f"\n[*] جاري جلب {limit} شمعة لـ {symbol}...")
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    return df

def add_technical_indicators(df):
    """إضافة مؤشرات فنية"""
    df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
    df['ema_9'] = talib.EMA(df['close'], timeperiod=9)
    df['ema_21'] = talib.EMA(df['close'], timeperiod=21)
    df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['volume_sma_20'] = talib.SMA(df['volume'], timeperiod=20)
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    return df

def entry_signal(row, prev_row):
    """إشارة دخول محسنة"""
    if pd.isna(row['rsi_14']) or pd.isna(row['macd']):
        return False
    
    score = 0
    
    if row['ema_9'] > row['ema_21']:
        score += 2
    
    if row['macd'] > row['macd_signal'] and row['macd_hist'] > 0:
        score += 2
    
    if 30 < row['rsi_14'] < 65:
        score += 1
    elif row['rsi_14'] <= 30:
        score += 3
    
    if row['bb_position'] < 0.5:
        score += 2
    
    if row['volume_ratio'] > 1.2:
        score += 2
    
    if prev_row is not None and row['close'] > prev_row['close'] and prev_row['close'] > prev_row['open']:
        score += 1
    
    return score >= 7

def backtest_with_params(df, tp_pct, sl_pct, initial_capital=10000):
    """اختبار الاستراتيجية مع معايير خروج محددة"""
    capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    in_position = False
    
    df_copy = df.copy()
    df_copy['prev_close'] = df_copy['close'].shift(1)
    df_copy['prev_open'] = df_copy['open'].shift(1)
    
    for i in range(50, len(df_copy) - 10):
        row = df_copy.iloc[i]
        prev_row = df_copy.iloc[i-1] if i > 0 else None
        
        if in_position:
            high = row['high']
            low = row['low']
            
            # فحص Take Profit أولاً
            if high >= entry_price * (1 + tp_pct):
                exit_price = entry_price * (1 + tp_pct)
                pnl_pct = tp_pct
                capital += position * (exit_price - entry_price)
                trades.append({'entry': entry_price, 'exit': exit_price, 'pnl_pct': pnl_pct, 'result': 'WIN'})
                in_position = False
                position = 0
            # فحص Stop Loss
            elif low <= entry_price * (1 - sl_pct):
                exit_price = entry_price * (1 - sl_pct)
                pnl_pct = -sl_pct
                capital += position * (exit_price - entry_price)
                trades.append({'entry': entry_price, 'exit': exit_price, 'pnl_pct': pnl_pct, 'result': 'LOSS'})
                in_position = False
                position = 0
        
        if not in_position:
            if entry_signal(row, prev_row):
                risk_amount = capital * 0.02
                stop_loss_distance = row['close'] * sl_pct
                if stop_loss_distance > 0:
                    position_size = risk_amount / stop_loss_distance
                    position = position_size
                    entry_price = row['close']
                    in_position = True
    
    if in_position and len(df_copy) > 0:
        final_price = df_copy.iloc[-1]['close']
        pnl_pct = (final_price - entry_price) / entry_price
        capital += position * (final_price - entry_price)
        trades.append({'entry': entry_price, 'exit': final_price, 'pnl_pct': pnl_pct, 'result': 'CLOSED'})
    
    return capital, trades

# اختبار على BTC
symbol = 'BTC/USDT:USDT'
df = fetch_ohlcv(symbol, timeframe='1h', limit=5000)
df = add_technical_indicators(df)
df.dropna(inplace=True)

print(f"\n{'='*70}")
print(f"اختبار معايير الخروج المختلفة على {symbol}")
print('='*70)

tp_values = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
sl_values = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]

results = []

print(f"\nجاري اختبار {len(tp_values) * len(sl_values)} تركيبة من TP/SL...\n")

for tp in tp_values:
    for sl in sl_values:
        final_capital, trades = backtest_with_params(df, tp, sl)
        
        total_trades = len(trades)
        if total_trades == 0:
            continue
            
        winning_trades = [t for t in trades if t['result'] == 'WIN']
        losing_trades = [t for t in trades if t['result'] == 'LOSS']
        
        win_rate = len(winning_trades) / total_trades * 100
        avg_win = np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl_pct'] for t in losing_trades]) if losing_trades else 0
        total_return = (final_capital - 10000) / 10000 * 100
        
        gross_profit = sum([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
        gross_loss = abs(sum([t['pnl_pct'] for t in losing_trades])) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        results.append({
            'tp': tp,
            'sl': sl,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_return': total_return,
            'final_capital': final_capital,
            'profit_factor': profit_factor
        })

# تحويل النتائج إلى DataFrame
results_df = pd.DataFrame(results)

# ترتيب حسب العائد الكلي
results_df_sorted = results_df.sort_values('total_return', ascending=False)

print(f"\n{'='*70}")
print("أفضل 10 تركيبات من TP/SL")
print('='*70)
print(f"{'TP':<8} {'SL':<8} {'صفقات':<8} {'Win%':<8} {'AvgWin':<10} {'AvgLoss':<10} {'العائد%':<10} {'PF':<8}")
print('-'*70)

for _, row in results_df_sorted.head(10).iterrows():
    print(f"{row['tp']*100:.1f}%    {row['sl']*100:.1f}%    {row['total_trades']:<8} {row['win_rate']:.1f}%     {row['avg_win']:.4f}     {row['avg_loss']:.4f}     {row['total_return']:+.2f}%    {row['profit_factor']:.2f}")

# البحث عن أفضل تركيبة
best = results_df_sorted.iloc[0]
print(f"\n{'='*70}")
print("🏆 أفضل تركيبة:")
print('='*70)
print(f"Take Profit: {best['tp']*100:.1f}%")
print(f"Stop Loss: {best['sl']*100:.1f}%")
print(f"عدد الصفقات: {best['total_trades']}")
print(f"نسبة الربح: {best['win_rate']:.1f}%")
print(f"العائد الكلي: {best['total_return']:+.2f}%")
print(f"رأس المال النهائي: ${best['final_capital']:.2f}")
print(f"Profit Factor: {best['profit_factor']:.2f}")

# تحليل إضافي
print(f"\n{'='*70}")
print("تحليل نسبة المخاطرة/العائد (Risk/Reward)")
print('='*70)

for tp in [0.02, 0.03, 0.04]:
    for sl in [0.01, 0.015, 0.02]:
        rr_ratio = tp / sl
        subset = results_df[(results_df['tp'] == tp) & (results_df['sl'] == sl)]
        if len(subset) > 0:
            avg_return = subset['total_return'].values[0]
            print(f"R:R = {rr_ratio:.2f} (TP={tp*100:.1f}%, SL={sl*100:.1f}%) -> العائد: {avg_return:+.2f}%")

print(f"\n{'='*70}")
print("التوصيات النهائية")
print('='*70)

if best['total_return'] > 0 and best['profit_factor'] > 1.2:
    print(f"✅ النتائج إيجابية! استخدم TP={best['tp']*100:.1f}% و SL={best['sl']*100:.1f}%")
elif best['total_return'] > -10:
    print(f"⚠️ النتائج قريبة من التعادل. قد تحتاج لتحسين إشارات الدخول.")
else:
    print("❌ النتائج سلبية حتى مع أفضل معايير الخروج.")
    print("   المشكلة الرئيسية في إشارات الدخول وليست في الخروج.")
    print("   يجب تحسين فلتر الدخول أو إضافة المزيد من الشروط.")

print("\nملاحظات مهمة:")
print("- Profit Factor > 1.5 يعتبر جيد")
print("- Win Rate > 45% يعتبر مقبول")
print("- العائد الإيجابي مع PF > 1.2 هو الهدف الأساسي")
print("- قد تحتاج لتجربة أطر زمنية مختلفة (4H, 1D)")

print("\n" + "="*70)
print("تم الانتهاء من تحسين معايير الخروج")
print("="*70)
