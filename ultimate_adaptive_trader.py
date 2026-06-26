import pandas as pd
import numpy as np
import ccxt

class UltimateAdaptiveTrader:
    """
    نظام تداول احترافي متكيف مع جميع ظروف السوق
    - يستخدم Long و Short حسب الاتجاه
    - فلترة ذكية للصفقات
    - إدارة مخاطر ديناميكية متقدمة
    """
    
    def __init__(self, symbol='BTC/USDT', timeframe='4h'):
        self.symbol = symbol
        self.timeframe = timeframe  # فريم 4 ساعات أفضل للتداول المتوسط
        self.exchange = ccxt.binance()
        self.df = None
        self.capital = 10000
        self.position = None
        self.trades = []
        
    def fetch_data(self, limit=5000):
        """جلب بيانات أطول"""
        print(f"📡 جاري جلب بيانات {self.symbol} على فريم {self.timeframe}...")
        bars = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        self.df = df
        return df

    def add_indicators(self):
        """إضافة مؤشرات احترافية"""
        df = self.df
        
        # VWAP
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['cum_vol'] = df['volume'].cumsum()
        df['cum_tp_vol'] = (df['typical_price'] * df['volume']).cumsum()
        df['vwap'] = df['cum_tp_vol'] / df['cum_vol']
        
        # EMA للاتجاه
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_mid'] + (bb_std * 2)
        df['bb_lower'] = df['bb_mid'] - (bb_std * 2)
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # ADX
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        tr = true_range
        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(14).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        self.df = df.dropna()
        return self.df

    def detect_trend(self, row):
        """تحديد الاتجاه الرئيسي بدقة"""
        close = row['close']
        ema_20 = row['ema_20']
        ema_50 = row['ema_50']
        ema_200 = row['ema_200']
        adx = row['adx']
        
        # اتجاه قوي صاعد
        if close > ema_20 > ema_50 > ema_200 and row['plus_di'] > row['minus_di'] and adx > 20:
            return 'STRONG_BULL'
        # اتجاه ضعيف صاعد
        elif close > ema_200 and ema_20 > ema_50:
            return 'WEAK_BULL'
        # اتجاه قوي هابط
        elif close < ema_20 < ema_50 < ema_200 and row['minus_di'] > row['plus_di'] and adx > 20:
            return 'STRONG_BEAR'
        # اتجاه ضعيف هابط
        elif close < ema_200 and ema_20 < ema_50:
            return 'WEAK_BEAR'
        else:
            return 'SIDEWAYS'

    def get_signal(self, row, trend):
        """استراتيجية تداول ذكية حسب الاتجاه - نسخة محسنة"""
        signal = 'WAIT'
        reason = ''
        
        # في السوق الصاعد القوي: نبحث عن فرص شراء فقط
        if trend == 'STRONG_BULL':
            if row['rsi'] < 50 and row['macd'] > row['signal_line']:
                signal = 'LONG'
                reason = 'Pullback in Strong Bull'
            elif row['close'] > row['ema_20'] and row['rsi'] > 45 and row['rsi'] < 70:
                signal = 'LONG'
                reason = 'Momentum Continuation'
                
        # في السوق الصاعد الضعيف: حذر
        elif trend == 'WEAK_BULL':
            if row['rsi'] < 45 and row['close'] <= row['bb_lower'] * 1.03:
                signal = 'LONG'
                reason = 'Oversold in Weak Bull'
                
        # في السوق الهابط القوي: نبحث عن فرص بيع فقط
        elif trend == 'STRONG_BEAR':
            if row['rsi'] > 45 and row['macd'] < row['signal_line']:
                signal = 'SHORT'
                reason = 'Rally in Strong Bear'
            elif row['close'] < row['ema_20'] and row['rsi'] < 55 and row['rsi'] > 30:
                signal = 'SHORT'
                reason = 'Momentum Down'
                
        # في السوق الهابط الضعيف: حذر
        elif trend == 'WEAK_BEAR':
            if row['rsi'] > 60 and row['close'] >= row['bb_upper'] * 0.97:
                signal = 'SHORT'
                reason = 'Overbought in Weak Bear'
                
        # سوق جانبي: تداول بين الحدود
        elif trend == 'SIDEWAYS':
            if row['rsi'] < 40 and row['close'] <= row['bb_lower'] * 1.02:
                signal = 'LONG'
                reason = 'Range Bottom'
            elif row['rsi'] > 60 and row['close'] >= row['bb_upper'] * 0.98:
                signal = 'SHORT'
                reason = 'Range Top'
                
        return signal, reason

    def dynamic_risk(self, row, signal):
        """إدارة مخاطر متقدمة"""
        atr = row['atr']
        close = row['close']
        
        # نسبة مخاطرة مختلفة حسب قوة الاتجاه
        if 'STRONG' in self.detect_trend(row):
            sl_multiplier = 2.0
            tp_multiplier = 3.5
        else:
            sl_multiplier = 2.5
            tp_multiplier = 3.0
            
        if signal == 'LONG':
            stop_loss = close - (sl_multiplier * atr)
            take_profit = close + (tp_multiplier * atr)
        elif signal == 'SHORT':
            stop_loss = close + (sl_multiplier * atr)
            take_profit = close - (tp_multiplier * atr)
        else:
            stop_loss, take_profit = None, None
            
        return stop_loss, take_profit

    def run_backtest(self):
        """تشغيل الباك تست"""
        print("🧠 جاري تشغيل الاستراتيجية الاحترافية...")
        df = self.df
        capital = self.capital
        position = None
        trades = []
        
        for i in range(350, len(df)):
            row = df.iloc[i]
            current_price = row['close']
            
            trend = self.detect_trend(row)
            signal, reason = self.get_signal(row, trend)
            
            if position is None:
                if signal in ['LONG', 'SHORT']:
                    sl, tp = self.dynamic_risk(row, signal)
                    position = {
                        'entry_price': current_price,
                        'stop_loss': sl,
                        'take_profit': tp,
                        'entry_time': row.name,
                        'type': signal,
                        'trend': trend,
                        'reason': reason
                    }
            else:
                should_close = False
                exit_reason = ''
                
                if position['type'] == 'LONG':
                    if current_price <= position['stop_loss']:
                        should_close = True
                        exit_reason = 'SL'
                    elif current_price >= position['take_profit']:
                        should_close = True
                        exit_reason = 'TP'
                    # خروج إذا انعكس الاتجاه
                    elif self.detect_trend(row) == 'STRONG_BEAR':
                        should_close = True
                        exit_reason = 'TREND_REVERSAL'
                        
                elif position['type'] == 'SHORT':
                    if current_price >= position['stop_loss']:
                        should_close = True
                        exit_reason = 'SL'
                    elif current_price <= position['take_profit']:
                        should_close = True
                        exit_reason = 'TP'
                    # خروج إذا انعكس الاتجاه
                    elif self.detect_trend(row) == 'STRONG_BULL':
                        should_close = True
                        exit_reason = 'TREND_REVERSAL'
                
                if should_close:
                    exit_price = current_price
                    if position['type'] == 'LONG':
                        pnl_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
                    else:
                        pnl_pct = ((position['entry_price'] - exit_price) / position['entry_price']) * 100
                        
                    pnl_amount = (capital * pnl_pct) / 100
                    capital += pnl_amount
                    
                    trades.append({
                        'entry': position['entry_time'],
                        'exit': row.name,
                        'pnl_pct': pnl_pct,
                        'pnl_amount': pnl_amount,
                        'type': position['type'],
                        'trend': position['trend'],
                        'reason': exit_reason
                    })
                    position = None
                    
                # Trailing Stop
                if position:
                    if position['type'] == 'LONG':
                        new_sl = current_price - (2.0 * row['atr'])
                        if new_sl > position['stop_loss']:
                            position['stop_loss'] = new_sl
                    elif position['type'] == 'SHORT':
                        new_sl = current_price + (2.0 * row['atr'])
                        if new_sl < position['stop_loss']:
                            position['stop_loss'] = new_sl

        # الإحصائيات
        total_trades = len(trades)
        winning_trades = [t for t in trades if t['pnl_pct'] > 0]
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
        total_pnl = sum(t['pnl_amount'] for t in trades)
        final_capital = self.capital + total_pnl
        roi = ((final_capital - self.capital) / self.capital) * 100
        
        long_trades = [t for t in trades if t['type'] == 'LONG']
        short_trades = [t for t in trades if t['type'] == 'SHORT']
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'final_capital': final_capital,
            'roi': roi,
            'trades': trades,
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'long_win_rate': (len([t for t in long_trades if t['pnl_pct'] > 0]) / len(long_trades) * 100) if long_trades else 0,
            'short_win_rate': (len([t for t in short_trades if t['pnl_pct'] > 0]) / len(short_trades) * 100) if short_trades else 0,
        }

    def print_report(self, results):
        print("\n" + "="*70)
        print("🚀 تقرير أداء النظام الاحترافي المتكيف (Ultimate Adaptive Trader)")
        print("="*70)
        print(f"💰 رأس المال الأولي: ${self.capital:,.2f}")
        print(f"🏦 رأس المال النهائي: ${results['final_capital']:,.2f}")
        print(f"📈 العائد على الاستثمار (ROI): {results['roi']:.2f}%")
        print(f"🔢 عدد الصفقات المنفذة: {results['total_trades']}")
        print(f"🎯 نسبة الفوز العامة: {results['win_rate']:.2f}%")
        print(f"💵 صافي الربح/الخسارة: ${results['total_pnl']:,.2f}")
        print("-"*70)
        print("📊 الأداء حسب نوع الصفقة:")
        print(f"   • صفقات LONG: {results['long_trades']} | نسبة فوز: {results['long_win_rate']:.1f}%")
        print(f"   • صفقات SHORT: {results['short_trades']} | نسبة فوز: {results['short_win_rate']:.1f}%")
        print("="*70)
        
        if results['roi'] > 20:
            print("✅ النتيجة ممتازة! النظام يحقق أرباحاً قوية.")
        elif results['roi'] > 0:
            print("✅ النتيجة جيدة! النظام ربحي ويحتاج للمزيد من التحسين.")
        elif results['roi'] > -10:
            print("⚠️ النتيجة مقبولة نسبياً، تحتاج لضبط المعاملات.")
        else:
            print("❌ النتيجة ضعيفة، يجب مراجعة الاستراتيجية.")
        print("="*70)

# --- التشغيل ---
if __name__ == "__main__":
    trader = UltimateAdaptiveTrader(symbol='BTC/USDT', timeframe='4h')
    
    try:
        trader.fetch_data(limit=5000)
        trader.add_indicators()
        results = trader.run_backtest()
        trader.print_report(results)
        
        if results['trades']:
            print("\n📋 آخر 5 صفقات منفذة:")
            for t in results['trades'][-5:]:
                status = "ربح ✅" if t['pnl_pct'] > 0 else "خسارة ❌"
                print(f"- {t['entry'].date()} | {t['type']} | {t['trend']} | {status} | {t['pnl_pct']:.2f}% ({t['reason']})")
                
    except Exception as e:
        print(f"حدث خطأ: {e}")
