#!/usr/bin/env python3
"""
متقدم: تحسين استراتيجية الدخول مع فلاتر قوية
يضيف: RSI, MACD, Bollinger Bands, Volume Filters, Volatility Filter
"""

import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import talib
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("نظام التداول المتقدم - إضافة فلاتر الدخول القوية")
print("="*70)

# 1. جلب البيانات الموسعة (50,000 شمعة)
def fetch_ohlcv(symbol, timeframe='1h', limit=50000):
    """جلب بيانات OHLCV مع مؤشرات فنية متقدمة"""
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
    """إضافة مؤشرات فنية متقدمة"""
    # RSI
    df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
    df['rsi_7'] = talib.RSI(df['close'], timeperiod=7)
    
    # MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
        df['close'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    
    # Bollinger Bands
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
        df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )
    
    # EMA
    df['ema_9'] = talib.EMA(df['close'], timeperiod=9)
    df['ema_21'] = talib.EMA(df['close'], timeperiod=21)
    df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
    
    # ATR (لتقلب السوق)
    df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    
    # Volume indicators
    df['volume_sma_20'] = talib.SMA(df['volume'], timeperiod=20)
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    # Price position relative to BB
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Trend strength
    df['trend_strength'] = abs(df['ema_9'] - df['ema_50']) / df['close']
    
    return df

def create_advanced_features(df):
    """إنشاء ميزات متقدمة للنموذج"""
    features = df.copy()
    
    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        features[f'return_lag_{lag}'] = features['close'].pct_change(lag)
        features[f'rsi_lag_{lag}'] = features['rsi_14'].shift(lag)
        features[f'macd_lag_{lag}'] = features['macd'].shift(lag)
    
    # Rolling statistics
    for window in [5, 10, 20]:
        features[f'return_mean_{window}'] = features['close'].pct_change().rolling(window).mean()
        features[f'return_std_{window}'] = features['close'].pct_change().rolling(window).std()
        features[f'rsi_mean_{window}'] = features['rsi_14'].rolling(window).mean()
        features[f'volume_mean_{window}'] = features['volume_ratio'].rolling(window).mean()
    
    # Cross indicators
    features['ema_cross'] = (features['ema_9'] > features['ema_21']).astype(int)
    features['macd_cross'] = (features['macd'] > features['macd_signal']).astype(int)
    features['rsi_oversold'] = (features['rsi_14'] < 30).astype(int)
    features['rsi_overbought'] = (features['rsi_14'] > 70).astype(int)
    features['bb_squeeze'] = (features['bb_position'] < 0.2).astype(int)
    features['bb_expansion'] = (features['bb_position'] > 0.8).astype(int)
    
    # Volatility adjusted returns
    features['vol_adj_return'] = features['close'].pct_change() / (features['atr_14'] / features['close'])
    
    return features

def create_target(df, lookahead=5, threshold=0.01):
    """إنشاء الهدف: 1 إذا كان العائد المستقبلي > threshold، وإلا 0"""
    future_return = df['close'].shift(-lookahead) / df['close'] - 1
    target = (future_return > threshold).astype(int)
    return target

def advanced_entry_filter(df, row_idx):
    """فلتر دخول متقدم بشروط متعددة"""
    if row_idx >= len(df) - 5:  # تجنب آخر 5 صفوف
        return False, 0
    
    row = df.iloc[row_idx]
    score = 0
    reasons = []
    
    # شرط 1: اتجاه إيجابي (EMA 9 > EMA 21)
    if row['ema_9'] > row['ema_21']:
        score += 2
        reasons.append("EMA Bullish")
    
    # شرط 2: MACD إيجابي
    if row['macd'] > row['macd_signal'] and row['macd_hist'] > 0:
        score += 2
        reasons.append("MACD Bullish")
    
    # شرط 3: RSI في منطقة معقولة (ليس تشبع شرائي)
    if 30 < row['rsi_14'] < 65:
        score += 1
        reasons.append("RSI Neutral")
    elif row['rsi_14'] <= 30:  # تشبع بيعي - فرصة شراء
        score += 3
        reasons.append("RSI Oversold")
    
    # شرط 4: السعر قريب من الحد السفلي لبولينجر أو في منتصف
    if row['bb_position'] < 0.5:
        score += 2
        reasons.append("BB Position Good")
    
    # شرط 5: حجم تداول أعلى من المتوسط
    if row['volume_ratio'] > 1.2:
        score += 2
        reasons.append("High Volume")
    
    # شرط 6: تقلب معقول (ليس عالي جداً)
    avg_atr = df['atr_14'].iloc[max(0, row_idx-20):row_idx].mean()
    if row['atr_14'] < avg_atr * 1.5:
        score += 1
        reasons.append("Normal Volatility")
    
    # شرط 7: زخم إيجابي
    if row['return_lag_1'] > 0 and row['return_lag_2'] > 0:
        score += 1
        reasons.append("Positive Momentum")
    
    # قرار الدخول: درجة >= 8
    enter = score >= 8
    return enter, score

def backtest_advanced_strategy(df, initial_capital=10000, tp_pct=0.03, sl_pct=0.015):
    """اختبار الاستراتيجية المتقدمة"""
    capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    
    in_position = False
    
    for i in range(len(df) - 10):  # ترك مساحة للـ lookahead
        row = df.iloc[i]
        
        # فحص الخروج إذا كنا في مركز
        if in_position:
            current_price = row['close']
            pnl_pct = (current_price - entry_price) / entry_price
            
            # Take Profit
            if pnl_pct >= tp_pct:
                capital += position * (current_price - entry_price)
                trades.append({'entry': entry_price, 'exit': current_price, 'pnl_pct': pnl_pct, 'result': 'WIN'})
                in_position = False
                position = 0
            # Stop Loss
            elif pnl_pct <= -sl_pct:
                capital += position * (current_price - entry_price)
                trades.append({'entry': entry_price, 'exit': current_price, 'pnl_pct': pnl_pct, 'result': 'LOSS'})
                in_position = False
                position = 0
        
        # فحص الدخول إذا لم نكن في مركز
        if not in_position:
            enter, score = advanced_entry_filter(df, i)
            if enter and score >= 8:
                # حساب حجم المركز (2% من رأس المال مخاطرة)
                risk_amount = capital * 0.02
                stop_loss_distance = entry_price * sl_pct if (entry_price := row['close']) else 0
                if stop_loss_distance > 0:
                    position_size = risk_amount / stop_loss_distance
                    position = position_size
                    entry_price = row['close']
                    in_position = True
    
    # إغلاق أي مركز مفتوح في النهاية
    if in_position and len(df) > 0:
        final_price = df.iloc[-1]['close']
        pnl_pct = (final_price - entry_price) / entry_price
        capital += position * (final_price - entry_price)
        trades.append({'entry': entry_price, 'exit': final_price, 'pnl_pct': pnl_pct, 'result': 'CLOSED'})
    
    return capital, trades

# === التنفيذ الرئيسي ===
symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
results = {}

for symbol in symbols:
    print(f"\n{'='*70}")
    print(f"معالجة {symbol}")
    print('='*70)
    
    # جلب البيانات
    df = fetch_ohlcv(symbol, timeframe='1h', limit=50000)
    print(f"✓ تم جلب {len(df)} شمعة")
    
    # إضافة المؤشرات
    df = add_technical_indicators(df)
    print("✓ تمت إضافة المؤشرات الفنية")
    
    # إنشاء الميزات
    df_features = create_advanced_features(df)
    df_features.dropna(inplace=True)
    print(f"✓ تم إنشاء {len(df_features.columns)} ميزة")
    
    # إنشاء الهدف
    df_features['target'] = create_target(df_features, lookahead=5, threshold=0.015)
    df_features.dropna(inplace=True)
    
    # تحضير البيانات للتدريب
    feature_cols = [col for col in df_features.columns if col not in ['target', 'open', 'high', 'low', 'close', 'volume']]
    X = df_features[feature_cols]
    y = df_features['target']
    
    # تقسيم البيانات
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # تدريب نموذج Gradient Boosting
    print("\n[*] جاري تدريب نموذج Gradient Boosting...")
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=50,
        min_samples_leaf=20,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # التقييم
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✓ دقة النموذج: {accuracy:.2%}")
    
    # اختبار الاستراتيجية
    print("\n[*] جاري اختبار الاستراتيجية المتقدمة...")
    df_with_signals = df_features.copy()
    df_with_signals['signal'] = 0
    df_with_signals['score'] = 0
    
    for i in range(len(df_with_signals)):
        enter, score = advanced_entry_filter(df_with_signals, i)
        if enter:
            df_with_signals.loc[df_with_signals.index[i], 'signal'] = 1
            df_with_signals.loc[df_with_signals.index[i], 'score'] = score
    
    final_capital, trades = backtest_advanced_strategy(df_with_signals, initial_capital=10000, tp_pct=0.03, sl_pct=0.015)
    
    # تحليل النتائج
    total_trades = len(trades)
    winning_trades = [t for t in trades if t['result'] == 'WIN']
    losing_trades = [t for t in trades if t['result'] == 'LOSS']
    
    win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
    avg_win = np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t['pnl_pct'] for t in losing_trades]) if losing_trades else 0
    total_return = (final_capital - 10000) / 10000 * 100
    
    profit_factor = abs(sum([t['pnl_pct'] for t in winning_trades]) / sum([t['pnl_pct'] for t in losing_trades])) if losing_trades and sum([t['pnl_pct'] for t in losing_trades]) != 0 else float('inf')
    
    results[symbol] = {
        'accuracy': accuracy,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'total_return': total_return,
        'final_capital': final_capital,
        'profit_factor': profit_factor
    }
    
    print(f"\n{'='*70}")
    print(f"نتائج {symbol}")
    print('='*70)
    print(f"عدد الصفقات: {total_trades}")
    print(f"نسبة الربح: {win_rate:.2f}%")
    print(f"متوسط الربح: {avg_win:.4f}")
    print(f"متوسط الخسارة: {avg_loss:.4f}")
    print(f"العائد الكلي: {total_return:.2f}%")
    print(f"رأس المال النهائي: ${final_capital:.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")

# ملخص عام
print(f"\n{'='*70}")
print("الملخص العام")
print('='*70)
print(f"{'العملة':<15} {'الدقة':<10} {'الصفقات':<10} {'Win Rate':<10} {'العائد%':<10} {'Profit Factor':<15}")
print('-'*70)
for symbol, res in results.items():
    print(f"{symbol:<15} {res['accuracy']:.2%}      {res['total_trades']:<10} {res['win_rate']:.2f}%      {res['total_return']:+.2f}%       {res['profit_factor']:.2f}")

# التوصيات
print(f"\n{'='*70}")
print("التوصيات")
print('='*70)

avg_win_rate = np.mean([r['win_rate'] for r in results.values()])
avg_return = np.mean([r['total_return'] for r in results.values()])
avg_profit_factor = np.mean([r['profit_factor'] for r in results.values() if r['profit_factor'] != float('inf')])

if avg_win_rate > 45 and avg_return > 0 and avg_profit_factor > 1.2:
    print("✅ النتائج مشجعة! يمكن الانتقال للاختبار على الحساب التجريبي.")
elif avg_win_rate > 40 and avg_profit_factor > 0.9:
    print("⚠️ النتائج متوسطة. تحتاج لتحسين طفيف قبل الحساب التجريبي.")
else:
    print("❌ النتائج غير كافية. يجب إعادة النظر في الاستراتيجية أو المعايير.")

print("\nملاحظات:")
if avg_win_rate < 45:
    print("- نسبة الربح منخفضة، حاول تعديل فلتر الدخول ليكون أكثر انتقائية.")
if avg_return < 0:
    print("- العائد سلبي، جرب تعديل نسب TP/SL أو تحسين توقيت الدخول.")
if avg_profit_factor < 1.2:
    print("- Profit Factor منخفض، الخسائر كبيرة مقارنة بالأرباح.")

print("\n" + "="*70)
print("تم الانتهاء من التحليل المتقدم")
print("="*70)
