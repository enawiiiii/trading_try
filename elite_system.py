#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
نظام النخبة للتداول الآلي (Elite Trading System)
نظام متكامل يعتمد على 4 وكلاء ذكيين (Agents) يعملون معاً:
1. وكيل الاتجاه (Trend Agent): يحدد الاتجاه العام باستخدام EMA و ADX
2. وكيل الزخم (Momentum Agent): يقيس قوة الحركة باستخدام RSI و MACD
3. وكيل التقلب (Volatility Agent): يقيس المخاطرة باستخدام ATR و Bollinger Bands
4. مجلس الإدارة (Board Agent): يجمع الآراء ويتخذ القرار النهائي مع إدارة مخاطر ديناميكية

المميزات:
- استخدام VWAP لمحاكاة تداول المؤسسات
- وقف خسارة وجني أرباح ديناميكيان بناءً على التقلب
- فلترة صارمة للصفقات (لا دخول إلا بتوافق 3 وكلاء)
- تقرير أداء شامل مع تحليل المخاطر
"""

import pandas as pd
import numpy as np
import talib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class EliteTradingSystem:
    def __init__(self, symbol='BTCUSDT', timeframe='1d'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.data = None
        self.indicators = {}
        self.signals = []
        self.trades = []
        self.capital = 10000  # رأس مال افتراضي 10,000$
        self.current_capital = self.capital
        
    def fetch_data(self, limit=1000):
        """جلب البيانات من Binance (محاكاة للبيانات الحقيقية)"""
        print(f"📡 جاري جلب بيانات {self.symbol}...")
        # محاكاة بيانات حقيقية (في الواقع نستخدم API بينانس)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=limit, freq=self.timeframe)
        np.random.seed(42)
        
        # توليد سعر عشوائي واقعي (Random Walk with Drift)
        price = 50000
        prices = [price]
        volumes = [np.random.uniform(1000, 5000)]  # بدء الحجم بأول قيمة
        
        for i in range(limit-1):
            change = np.random.normal(0.001, 0.03)  # تغير يومي متوسط 0.1% وتقلب 3%
            price = price * (1 + change)
            prices.append(price)
            volumes.append(np.random.uniform(1000, 5000))
            
        self.data = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * np.random.uniform(1.01, 1.05) for p in prices],
            'low': [p * np.random.uniform(0.95, 0.99) for p in prices],
            'close': prices,
            'volume': volumes
        })
        self.data.set_index('date', inplace=True)
        print(f"✅ تم جلب {len(self.data)} شمعة بنجاح.")
        
    def calculate_institutional_indicators(self):
        """حساب مؤشرات المؤسسات المتقدمة"""
        print("🧮 جاري حساب مؤشرات المؤسسات...")
        df = self.data
        
        # 1. VWAP (Volume Weighted Average Price) - سعر المؤسسات
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # 2. EMA للاتجاه طويل وقصير المدى
        df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
        df['ema_200'] = talib.EMA(df['close'], timeperiod=200)
        
        # 3. ADX لقوة الاتجاه
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        # 4. RSI للزخم
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        
        # 5. MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # 6. Bollinger Bands للتقلب
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        
        # 7. ATR للتقلب الديناميكي
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # 8. حجم نسبي
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        self.data = df
        self.indicators = {
            'vwap': df['vwap'],
            'ema_50': df['ema_50'],
            'ema_200': df['ema_200'],
            'adx': df['adx'],
            'rsi': df['rsi'],
            'macd': df['macd'],
            'bb_upper': df['bb_upper'],
            'bb_lower': df['bb_lower'],
            'atr': df['atr']
        }
        print("✅ اكتمل حساب جميع المؤشرات.")
        
    def trend_agent(self, idx):
        """وكيل الاتجاه: يقرر بناءً على EMA و ADX"""
        df = self.data
        if idx < 200: return 0  # لا توجد بيانات كافية
        
        score = 0
        # شرط الاتجاه الصاعد
        if df['close'].iloc[idx] > df['ema_50'].iloc[idx] > df['ema_200'].iloc[idx]:
            score += 1
        # شرط قوة الاتجاه
        if df['adx'].iloc[idx] > 25:
            score += 1
        # شرط السعر فوق VWAP (مؤسسات تشتري)
        if df['close'].iloc[idx] > df['vwap'].iloc[idx]:
            score += 1
            
        return score  # 0-3
    
    def momentum_agent(self, idx):
        """وكيل الزخم: يقرر بناءً على RSI و MACD"""
        df = self.data
        score = 0
        
        # RSI في منطقة صاعدة لكن غير متشبعة
        if 40 < df['rsi'].iloc[idx] < 70:
            score += 1
        # MACD إيجابي
        if df['macd'].iloc[idx] > df['macd_signal'].iloc[idx]:
            score += 1
        # تقاطع MACD إيجابي
        if idx > 1 and df['macd_hist'].iloc[idx] > 0 and df['macd_hist'].iloc[idx-1] <= 0:
            score += 1
            
        return score  # 0-3
    
    def volatility_agent(self, idx):
        """وكيل التقلب: يقرر بناءً على ATR و Bollinger Bands"""
        df = self.data
        score = 0
        
        # تقلب معتدل (ليس عالي جداً ولا منخفض جداً)
        avg_atr = df['atr'].rolling(window=50).mean().iloc[idx]
        if 0.5 * avg_atr < df['atr'].iloc[idx] < 1.5 * avg_atr:
            score += 1
        # السعر قريب من الحد السفلي للبولنجر (فرصة شراء)
        if df['close'].iloc[idx] < df['bb_lower'].iloc[idx] * 1.01:
            score += 1
        # حجم تداول عالي
        if df['volume_ratio'].iloc[idx] > 1.2:
            score += 1
            
        return score  # 0-3
    
    def board_agent(self, idx):
        """مجلس الإدارة: يجمع آراء الوكلاء ويتخذ القرار"""
        trend_score = self.trend_agent(idx)
        momentum_score = self.momentum_agent(idx)
        volatility_score = self.volatility_agent(idx)
        
        total_score = trend_score + momentum_score + momentum_score + volatility_score
        max_score = 9
        
        # قرار الشراء: تحتاج 6 نقاط على الأقل (توافق 3 وكلاء على الأقل)
        if total_score >= 6:
            return 'BUY', total_score / max_score
        # قرار البيع: إذا كانت النقاط منخفضة جداً
        elif total_score <= 2:
            return 'SELL', total_score / max_score
        else:
            return 'WAIT', total_score / max_score
            
    def dynamic_risk_management(self, idx):
        """إدارة مخاطر ديناميكية بناءً على ATR"""
        df = self.data
        atr = df['atr'].iloc[idx]
        current_price = df['close'].iloc[idx]
        
        # وقف الخسارة: 2 * ATR تحت السعر
        stop_loss = current_price - (2 * atr)
        # جني الربح: 3 * ATR فوق السعر (نسبة 1:1.5)
        take_profit = current_price + (3 * atr)
        
        return stop_loss, take_profit
        
    def simulate_trading(self):
        """محاكاة التداول على البيانات التاريخية"""
        print("\n🚀 بدء محاكاة التداول الذكي...")
        self.trades = []
        position = None  # None, 'LONG'
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        
        for idx in range(200, len(self.data)):
            decision, confidence = self.board_agent(idx)
            current_price = self.data['close'].iloc[idx]
            
            # فتح صفقة شراء
            if decision == 'BUY' and confidence > 0.65 and position is None:
                sl, tp = self.dynamic_risk_management(idx)
                position = 'LONG'
                entry_price = current_price
                stop_loss = sl
                take_profit = tp
                shares = self.current_capital * 0.95 / entry_price  # استخدام 95% من الرأس المال
                
            # إغلاق الصفقة
            elif position == 'LONG':
                # تحقق من وقف الخسارة
                if current_price <= stop_loss:
                    pnl = (stop_loss - entry_price) * shares
                    self.trades.append({
                        'type': 'LOSS_SL',
                        'entry': entry_price,
                        'exit': stop_loss,
                        'pnl': pnl,
                        'confidence': confidence
                    })
                    self.current_capital += pnl
                    position = None
                    
                # تحقق من جني الربح
                elif current_price >= take_profit:
                    pnl = (take_profit - entry_price) * shares
                    self.trades.append({
                        'type': 'PROFIT_TP',
                        'entry': entry_price,
                        'exit': take_profit,
                        'pnl': pnl,
                        'confidence': confidence
                    })
                    self.current_capital += pnl
                    position = None
                    
                # خروج طارئ إذا انعكس الاتجاه بقوة
                elif decision == 'SELL' and confidence > 0.7:
                    pnl = (current_price - entry_price) * shares
                    self.trades.append({
                        'type': 'EXIT_SIGNAL',
                        'entry': entry_price,
                        'exit': current_price,
                        'pnl': pnl,
                        'confidence': confidence
                    })
                    self.current_capital += pnl
                    position = None
                    
        print(f"✅ اكتملت المحاكاة. عدد الصفقات: {len(self.trades)}")
        
    def generate_report(self):
        """توليد تقرير أداء شامل"""
        if not self.trades:
            print("❌ لا توجد صفقات لتحليلها.")
            return
            
        wins = [t for t in self.trades if t['pnl'] > 0]
        losses = [t for t in self.trades if t['pnl'] <= 0]
        
        total_pnl = sum(t['pnl'] for t in self.trades)
        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0
        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
        profit_factor = abs(sum([t['pnl'] for t in wins]) / sum([t['pnl'] for t in losses])) if losses else float('inf')
        
        roi = (self.current_capital - self.capital) / self.capital * 100
        
        print("\n" + "="*60)
        print("🏆 تقرير أداء نظام النخبة للتداول")
        print("="*60)
        print(f"💰 رأس المال الأولي: ${self.capital:,.2f}")
        print(f"💵 رأس المال النهائي: ${self.current_capital:,.2f}")
        print(f"📈 العائد على الاستثمار (ROI): {roi:.2f}%")
        print(f"📊 عدد الصفقات الكلي: {len(self.trades)}")
        print(f"✅ عدد الصفقات الرابحة: {len(wins)} ({win_rate:.1f}%)")
        print(f"❌ عدد الصفقات الخاسرة: {len(losses)} ({100-win_rate:.1f}%)")
        print(f"📉 متوسط الربح في الصفقة الرابحة: ${avg_win:,.2f}")
        print(f"📉 متوسط الخسارة في الصفقة الخاسرة: ${avg_loss:,.2f}")
        print(f"⚖️ معامل الربحية (Profit Factor): {profit_factor:.2f}")
        
        # تحليل أنواع الخروج
        sl_count = len([t for t in self.trades if t['type'] == 'LOSS_SL'])
        tp_count = len([t for t in self.trades if t['type'] == 'PROFIT_TP'])
        signal_count = len([t for t in self.trades if t['type'] == 'EXIT_SIGNAL'])
        
        print(f"\n🔍 تحليل أنواع الخروج:")
        print(f"   - خروج بوقف الخسارة (SL): {sl_count}")
        print(f"   - خروج بجني الربح (TP): {tp_count}")
        print(f"   - خروج بإشارة عكسية: {signal_count}")
        
        if roi > 0 and win_rate > 50:
            print("\n🌟 التقييم: ممتاز! النظام جاهز للتجربة على حساب حقيقي.")
        elif roi > 0:
            print("\n⚠️ التقييم: جيد، لكن يحتاج تحسين نسبة الفوز.")
        else:
            print("\n❌ التقييم: يحتاج إعادة ضبط المعاملات.")
        print("="*60)

def main():
    print("🤖启动 نظام النخبة للتداول الآلي...")
    system = EliteTradingSystem(symbol='BTCUSDT', timeframe='1h')
    
    # 1. جلب البيانات
    system.fetch_data(limit=2000)
    
    # 2. حساب المؤشرات
    system.calculate_institutional_indicators()
    
    # 3. محاكاة التداول
    system.simulate_trading()
    
    # 4. توليد التقرير
    system.generate_report()

if __name__ == "__main__":
    main()
