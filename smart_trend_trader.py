import pandas as pd
import numpy as np
import ccxt

class SmartTrendTrader:
    """
    نظام تداول ذكي يتكيف مع حالة السوق (صاعد/هابط/جانبي)
    - في السوق الصاعد: يركز على الشراء عند الانخفاضات
    - في السوق الهابط: يركز على البيع عند الارتفاعات أو الانتظار
    - في السوق الجانبي: يتداول بين الدعم والمقاومة
    """
    
    def __init__(self, symbol='BTC/USDT', timeframe='1h'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange = ccxt.binance()
        self.df = None
        self.capital = 10000
        self.position = None
        self.trades = []
        
    def fetch_data(self, limit=3000):
        """جلب بيانات طويلة المدى لتحليل الاتجاه"""
        print(f"📡 جاري جلب بيانات {self.symbol}...")
        bars = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        self.df = df
        return df

    def add_all_indicators(self):
        """إضافة جميع المؤشرات الفنية"""
        df = self.df
        
        # 1. VWAP
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['cum_vol'] = df['volume'].cumsum()
        df['cum_tp_vol'] = (df['typical_price'] * df['volume']).cumsum()
        df['vwap'] = df['cum_tp_vol'] / df['cum_vol']
        
        # 2. Moving Averages
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        
        # 3. RSI & MACD
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # 4. Bollinger Bands
        df['bb_mid'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_mid'] + (bb_std * 2)
        df['bb_lower'] = df['bb_mid'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        
        # 5. ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # 6. ADX (قوة الاتجاه)
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

    def detect_market_state(self, row):
        """تحديد حالة السوق: صاعد، هابط، أو جانبي"""
        # استخدام ADX و EMA لتحديد الحالة
        adx = row['adx']
        ema_50 = row['ema_50']
        ema_200 = row['ema_200']
        close = row['close']
        
        if adx < 25:
            return 'SIDEWAYS' # سوق جانبي (ضعيف الاتجاه)
        elif close > ema_50 > ema_200 and row['plus_di'] > row['minus_di']:
            return 'BULLISH' # سوق صاعد قوي
        elif close < ema_50 < ema_200 and row['minus_di'] > row['plus_di']:
            return 'BEARISH' # سوق هابط قوي
        else:
            return 'TRANSITIONAL' # مرحلة انتقالية

    def get_signals(self, row, market_state):
        """الحصول على إشارة تداول بناءً على حالة السوق"""
        signal = 'WAIT'
        reason = ''
        
        if market_state == 'BULLISH':
            # استراتيجيات الشراء في السوق الصاعد
            if row['rsi'] < 40 and row['close'] > row['ema_200']: # شراء عند الانخفاض
                signal = 'BUY'
                reason = 'Pullback in Bull Market'
            elif row['macd'] > row['signal_line'] and row['rsi'] > 50 and row['rsi'] < 70:
                signal = 'BUY'
                reason = 'Momentum Continuation'
                
        elif market_state == 'BEARISH':
            # استراتيجيات البيع أو الانتظار في السوق الهابط
            if row['rsi'] > 65 and row['close'] < row['ema_200']: # بيع عند الارتفاع
                signal = 'SELL'
                reason = 'Rally in Bear Market'
            # في السوق الهابط القوي، الأفضل الانتظار
            else:
                signal = 'WAIT'
                reason = 'Strong Downtrend - Stay Safe'
                
        elif market_state == 'SIDEWAYS':
            # التداول بين الدعم والمقاومة
            if row['close'] <= row['bb_lower'] * 1.01 and row['rsi'] < 35:
                signal = 'BUY'
                reason = 'Oversold in Range'
            elif row['close'] >= row['bb_upper'] * 0.99 and row['rsi'] > 65:
                signal = 'SELL'
                reason = 'Overbought in Range'
                
        return signal, reason

    def dynamic_risk_management(self, row, signal):
        """إدارة مخاطر ديناميكية حسب حالة السوق"""
        atr = row['atr']
        close = row['close']
        
        if signal == 'BUY':
            stop_loss = close - (2.5 * atr)
            take_profit = close + (3.5 * atr)
        elif signal == 'SELL':
            stop_loss = close + (2.5 * atr)
            take_profit = close - (3.5 * atr)
        else:
            stop_loss, take_profit = None, None
            
        return stop_loss, take_profit

    def run_backtest(self):
        """تشغيل محاكاة التداول الذكي المتكيف"""
        print("🧠 جاري تحليل حالة السوق وتشغيل الاستراتيجية المتكيفة...")
        df = self.df
        capital = self.capital
        position = None
        trades = []
        
        for i in range(300, len(df)):
            row = df.iloc[i]
            current_price = row['close']
            
            # تحديد حالة السوق
            market_state = self.detect_market_state(row)
            
            # الحصول على الإشارة
            signal, reason = self.get_signals(row, market_state)
            
            if position is None:
                if signal in ['BUY', 'SELL']:
                    sl, tp = self.dynamic_risk_management(row, signal)
                    position = {
                        'entry_price': current_price,
                        'stop_loss': sl,
                        'take_profit': tp,
                        'entry_time': row.name,
                        'type': signal,
                        'market_state': market_state,
                        'reason': reason
                    }
            else:
                should_close = False
                exit_reason = ''
                
                if position['type'] == 'BUY':
                    if current_price <= position['stop_loss']:
                        should_close = True
                        exit_reason = 'SL'
                    elif current_price >= position['take_profit']:
                        should_close = True
                        exit_reason = 'TP'
                    # خروج إذا انعكس السوق لهابط
                    elif market_state == 'BEARISH' and row['rsi'] < 45:
                        should_close = True
                        exit_reason = 'MARKET_REVERSAL'
                        
                elif position['type'] == 'SELL':
                    if current_price >= position['stop_loss']:
                        should_close = True
                        exit_reason = 'SL'
                    elif current_price <= position['take_profit']:
                        should_close = True
                        exit_reason = 'TP'
                    # خروج إذا انعكس السوق لصاعد
                    elif market_state == 'BULLISH' and row['rsi'] > 55:
                        should_close = True
                        exit_reason = 'MARKET_REVERSAL'
                
                if should_close:
                    exit_price = current_price
                    if position['type'] == 'BUY':
                        pnl_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
                    else: # SELL
                        pnl_pct = ((position['entry_price'] - exit_price) / position['entry_price']) * 100
                        
                    pnl_amount = (capital * pnl_pct) / 100
                    capital += pnl_amount
                    
                    trades.append({
                        'entry': position['entry_time'],
                        'exit': row.name,
                        'pnl_pct': pnl_pct,
                        'pnl_amount': pnl_amount,
                        'type': position['type'],
                        'market_state': position['market_state'],
                        'reason': exit_reason
                    })
                    position = None
                    
                # Trailing Stop
                elif position['type'] == 'BUY':
                    new_sl = current_price - (2.5 * row['atr'])
                    if new_sl > position['stop_loss']:
                        position['stop_loss'] = new_sl
                elif position['type'] == 'SELL':
                    new_sl = current_price + (2.5 * row['atr'])
                    if new_sl < position['stop_loss']:
                        position['stop_loss'] = new_sl

        # الإحصائيات
        total_trades = len(trades)
        winning_trades = [t for t in trades if t['pnl_pct'] > 0]
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
        total_pnl = sum(t['pnl_amount'] for t in trades)
        final_capital = self.capital + total_pnl
        roi = ((final_capital - self.capital) / self.capital) * 100
        
        # تحليل حسب حالة السوق
        bullish_trades = [t for t in trades if t['market_state'] == 'BULLISH']
        bearish_trades = [t for t in trades if t['market_state'] == 'BEARISH']
        sideways_trades = [t for t in trades if t['market_state'] == 'SIDEWAYS']
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'final_capital': final_capital,
            'roi': roi,
            'trades': trades,
            'bullish_trades': len(bullish_trades),
            'bearish_trades': len(bearish_trades),
            'sideways_trades': len(sideways_trades),
            'bullish_win_rate': (len([t for t in bullish_trades if t['pnl_pct'] > 0]) / len(bullish_trades) * 100) if bullish_trades else 0,
            'bearish_win_rate': (len([t for t in bearish_trades if t['pnl_pct'] > 0]) / len(bearish_trades) * 100) if bearish_trades else 0,
            'sideways_win_rate': (len([t for t in sideways_trades if t['pnl_pct'] > 0]) / len(sideways_trades) * 100) if sideways_trades else 0,
        }

    def print_report(self, results):
        """طباعة تقرير مفصل"""
        print("\n" + "="*60)
        print("🚀 تقرير أداء النظام الذكي المتكيف (Smart Trend Trader)")
        print("="*60)
        print(f"💰 رأس المال الأولي: ${self.capital:,.2f}")
        print(f"🏦 رأس المال النهائي: ${results['final_capital']:,.2f}")
        print(f"📈 العائد على الاستثمار (ROI): {results['roi']:.2f}%")
        print(f"🔢 عدد الصفقات المنفذة: {results['total_trades']}")
        print(f"🎯 نسبة الفوز العامة: {results['win_rate']:.2f}%")
        print(f"💵 صافي الربح/الخسارة: ${results['total_pnl']:,.2f}")
        print("-"*60)
        print("📊 الأداء حسب حالة السوق:")
        print(f"   • صفقات صاعدة (Bullish): {results['bullish_trades']} | نسبة فوز: {results['bullish_win_rate']:.1f}%")
        print(f"   • صفقات هابطة (Bearish): {results['bearish_trades']} | نسبة فوز: {results['bearish_win_rate']:.1f}%")
        print(f"   • صفقات جانبية (Sideways): {results['sideways_trades']} | نسبة فوز: {results['sideways_win_rate']:.1f}%")
        print("="*60)
        
        if results['win_rate'] > 55:
            print("✅ النتيجة ممتازة! النظام جاهز للمرحلة التجريبية.")
        elif results['win_rate'] > 48:
            print("⚠️ النتيجة مقبولة، تحتاج لبعض الضبط الدقيق.")
        else:
            print("❌ النتيجة ضعيفة، يجب تعديل الاستراتيجية.")
        print("="*60)

# --- التشغيل الرئيسي ---
if __name__ == "__main__":
    trader = SmartTrendTrader(symbol='BTC/USDT', timeframe='1h')
    
    try:
        trader.fetch_data(limit=3000)
        trader.add_all_indicators()
        results = trader.run_backtest()
        trader.print_report(results)
        
        if results['trades']:
            print("\n📋 آخر 5 صفقات منفذة:")
            for t in results['trades'][-5:]:
                status = "ربح ✅" if t['pnl_pct'] > 0 else "خسارة ❌"
                print(f"- {t['entry'].date()} | {t['type']} | {t['market_state']} | {status} | {t['pnl_pct']:.2f}% ({t['reason']})")
                
    except Exception as e:
        print(f"حدث خطأ أثناء التنفيذ: {e}")
