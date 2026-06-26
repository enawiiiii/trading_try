import pandas as pd
import numpy as np
import ccxt
from datetime import datetime
import time

class FastSmartTrader:
    def __init__(self, symbol='BTC/USDT', timeframe='1h'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange = ccxt.binance()
        self.df = None
        self.capital = 10000
        self.position = None
        self.trades = []
        
    def fetch_data(self, limit=2000):
        """جلب بيانات سريعة ومحسنة"""
        print(f"📡 جاري جلب بيانات {self.symbol}...")
        bars = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        self.df = df
        return df

    def add_institutional_features(self):
        """إضافة مؤشرات المؤسسات بسرعة فائقة"""
        df = self.df
        
        # 1. VWAP (مؤشر المؤسسات الرئيسي)
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['cum_vol'] = df['volume'].cumsum()
        df['cum_tp_vol'] = (df['typical_price'] * df['volume']).cumsum()
        df['vwap'] = df['cum_tp_vol'] / df['cum_vol']
        
        # 2. Moving Averages (الاتجاه)
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # 3. RSI & MACD (الزخم)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # 4. Bollinger Bands (التقلب)
        df['bb_mid'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_mid'] + (bb_std * 2)
        df['bb_lower'] = df['bb_mid'] - (bb_std * 2)
        
        # 5. ATR (لإدارة المخاطر الديناميكية)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # تنظيف البيانات من القيم الناقصة
        self.df = df.dropna()
        return self.df

    def dynamic_risk_management(self, row):
        """حساب وقف الخسارة وجني الربح ديناميكياً بناءً على التقلب"""
        atr = row['atr']
        close = row['close']
        
        # وقف خسارة ديناميكي: 2.5 * ATR (أوسع قليلاً لتجنب الضوضاء)
        stop_loss = close - (2.5 * atr)
        # جني ربح ديناميكي: 4 * ATR (نسبة مخاطرة 1:1.6 لزيادة فرص الربح)
        take_profit = close + (4.0 * atr)
        
        return stop_loss, take_profit

    def quality_filter(self, row):
        """فلتر الصفقات الرديئة: يتطلب توافق مؤشرين على الأقل مع عدم وجود إشارات سلبية قوية"""
        score = 0
        negative_signals = 0
        
        # 1. فلتر الاتجاه (Trend) - قوي جداً
        if row['close'] > row['ema_200'] and row['close'] > row['vwap']:
            score += 2 # نعطيه وزن أكبر
            
        # 2. فلتر الزخم (Momentum) - متوازن
        if 45 < row['rsi'] < 70 and row['macd'] > row['signal_line']:
            score += 2
            
        # 3. فلتر القوة والاختراق (Strength)
        if row['close'] > row['bb_mid']:
            score += 1
            
        # إشارات سلبية تمنع الدخول
        if row['rsi'] > 80: # تشبع شرائي خطير جداً
            negative_signals += 1
        if row['close'] < row['ema_200']: # اتجاه هابط طويل الأجل
            negative_signals += 1
            
        # ندخل فقط إذا كانت النقاط الإيجابية >= 3 ولا توجد أكثر من إشارة سلبية واحدة
        return score >= 3 and negative_signals <= 1

    def run_backtest(self):
        """تشغيل محاكاة التداول الذكي"""
        print("🧠 جاري تشغيل العقل الاصطناعي وتحليل السوق...")
        df = self.df
        capital = self.capital
        position = None
        trades = []
        
        # نبدأ من المؤشر 250 لضمان اكتمال جميع الحسابات
        for i in range(250, len(df)):
            row = df.iloc[i]
            current_price = row['close']
            
            # إذا لم يكن لدينا صفقة، نبحث عن دخول
            if position is None:
                if self.quality_filter(row):
                    sl, tp = self.dynamic_risk_management(row)
                    # دخول صفقة شراء
                    position = {
                        'entry_price': current_price,
                        'stop_loss': sl,
                        'take_profit': tp,
                        'entry_time': row.name,
                        'type': 'BUY'
                    }
            else:
                # إدارة الصفقة المفتوحة
                # فحص وقف الخسارة أو جني الربح
                if current_price <= position['stop_loss'] or current_price >= position['take_profit']:
                    # إغلاق الصفقة
                    exit_price = current_price
                    pnl_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
                    pnl_amount = (capital * pnl_pct) / 100
                    capital += pnl_amount
                    
                    trades.append({
                        'entry': position['entry_time'],
                        'exit': row.name,
                        'pnl_pct': pnl_pct,
                        'pnl_amount': pnl_amount,
                        'type': position['type'],
                        'reason': 'TP' if current_price >= position['take_profit'] else 'SL'
                    })
                    position = None
                    
                # تحديث وقف الخسارة المتحرك (Trailing Stop مبسط)
                # إذا ارتفع السعر، نرفع وقف الخسارة لحماية الأرباح
                elif position['type'] == 'BUY':
                    new_sl = current_price - (2.5 * row['atr'])
                    if new_sl > position['stop_loss']:
                        position['stop_loss'] = new_sl
                        
                # خروج إضافي: إذا انعكس MACD ضدنا نخرج فوراً
                elif position['type'] == 'BUY' and row['macd'] < row['signal_line']:
                    # خروج مبكر إذا ضعف الزخم بشكل كبير
                    if row['rsi'] < 45: # زخم ضعيف جداً
                        exit_price = current_price
                        pnl_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
                        pnl_amount = (capital * pnl_pct) / 100
                        capital += pnl_amount
                        
                        trades.append({
                            'entry': position['entry_time'],
                            'exit': row.name,
                            'pnl_pct': pnl_pct,
                            'pnl_amount': pnl_amount,
                            'type': position['type'],
                            'reason': 'MACD_REVERSAL'
                        })
                        position = None

        # حساب الإحصائيات النهائية
        total_trades = len(trades)
        winning_trades = [t for t in trades if t['pnl_pct'] > 0]
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
        total_pnl = sum(t['pnl_amount'] for t in trades)
        final_capital = self.capital + total_pnl
        roi = ((final_capital - self.capital) / self.capital) * 100
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'final_capital': final_capital,
            'roi': roi,
            'trades': trades
        }

    def print_report(self, results):
        """طباعة تقرير مفصل"""
        print("\n" + "="*50)
        print("🚀 تقرير أداء النظام الذكي (Fast Smart Trader)")
        print("="*50)
        print(f"💰 رأس المال الأولي: ${self.capital:,.2f}")
        print(f"🏦 رأس المال النهائي: ${results['final_capital']:,.2f}")
        print(f"📈 العائد على الاستثمار (ROI): {results['roi']:.2f}%")
        print(f"🔢 عدد الصفقات المنفذة: {results['total_trades']}")
        print(f"🎯 نسبة الفوز: {results['win_rate']:.2f}%")
        print(f"💵 صافي الربح/الخسارة: ${results['total_pnl']:,.2f}")
        print("="*50)
        
        if results['win_rate'] > 55:
            print("✅ النتيجة ممتازة! النظام جاهز للمرحلة التجريبية.")
        elif results['win_rate'] > 45:
            print("⚠️ النتيجة مقبولة، تحتاج لبعض الضبط الدقيق.")
        else:
            print("❌ النتيجة ضعيفة، يجب تعديل الفلتر أو المؤشرات.")
        print("="*50)

# --- التشغيل الرئيسي ---
if __name__ == "__main__":
    trader = FastSmartTrader(symbol='BTC/USDT', timeframe='1h')
    
    try:
        # 1. جلب البيانات
        trader.fetch_data(limit=3000) # جلب 3000 شمعة للتدريب
        
        # 2. إضافة المؤشرات الذكية
        trader.add_institutional_features()
        
        # 3. تشغيل الباك تست
        results = trader.run_backtest()
        
        # 4. عرض التقرير
        trader.print_report(results)
        
        # عرض آخر 5 صفقات للتدقيق
        if results['trades']:
            print("\n📋 آخر 5 صفقات منفذة:")
            for t in results['trades'][-5:]:
                status = "ربح ✅" if t['pnl_pct'] > 0 else "خسارة ❌"
                print(f"- {t['entry'].date()} | {status} | الربح: {t['pnl_pct']:.2f}% ({t['reason']})")
                
    except Exception as e:
        print(f"حدث خطأ أثناء التنفيذ: {e}")
