import pandas as pd
import numpy as np
import ccxt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands
import warnings
warnings.filterwarnings('ignore')

class SimpleProfitableTrader:
    """
    نظام تداول بسيط وفعّال - قاعدة ذهبية:
    "اشترِ في الاتجاه الصاعد عندما يكون السعر منخفضاً نسبياً"
    """
    
    def __init__(self, symbol='BTC/USDT', timeframe='4h'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange = ccxt.binance()
        self.df = None
        
        print(f"🤖 نظام التداول البسيط لـ {symbol}...")

    def fetch_data(self, days=365):
        """جلب البيانات"""
        print("📡 جلب البيانات...")
        bars = days * 6 if self.timeframe == '4h' else days * 24
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=bars)
        self.df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], unit='ms')
        self.df.set_index('timestamp', inplace=True)
        print(f"✅ {len(self.df)} شمعة")
        return self.df

    def add_indicators(self):
        """إضافة مؤشرات بسيطة وفعّالة"""
        df = self.df.copy()
        
        # 1. الاتجاه العام (فوق SMA 200 = صاعد)
        df['sma_200'] = SMAIndicator(df['close'], window=200).sma_indicator()
        df['uptrend'] = (df['close'] > df['sma_200']).astype(int)
        
        # 2. RSI لتحديد مناطق التشبع
        df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
        
        # 3. Bolinger Bands للشراء عند الحدود السفلية
        bb = BollingerBands(df['close'])
        df['bb_low'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_high'] = bb.bollinger_hband()
        
        # 4. موقع السعر بالنسبة لبولينجر
        df['bb_pos'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
        
        # ====== استراتيجية الشراء البسيطة ======
        # اشترِ إذا كان:
        # 1. الاتجاه صاعد (السعر فوق SMA 200)
        # 2. RSI بين 40-60 (ليس متطرف)
        # 3. السعر في النصف السفلي من بولينجر (< 0.5)
        
        df['buy_signal'] = (
            (df['uptrend'] == 1) &
            (df['rsi'] > 40) & (df['rsi'] < 60) &
            (df['bb_pos'] < 0.5)
        ).astype(int)
        
        df.dropna(inplace=True)
        self.df = df
        print("🧠 المؤشرات جاهزة")
        return df

    def run_backtest(self, hold_period=5, stop_loss=0.03, take_profit=0.06):
        """باك تست للاستراتيجية البسيطة"""
        df = self.df.copy()
        
        capital = 10000
        current_capital = capital
        trades = []
        peak = capital
        max_dd = 0
        
        print("\n🚀 بدء المحاكاة...")
        print(f"📊 فترة الثبات: {hold_period} شموع")
        print(f"🛑 وقف الخسارة: {stop_loss*100}%")
        print(f"🎯 جني الربح: {take_profit*100}%")
        
        i = 0
        while i < len(df) - hold_period:
            row = df.iloc[i]
            
            if row['buy_signal'] == 1:
                entry = row['close']
                exit_idx = i + hold_period
                
                if exit_idx >= len(df):
                    break
                
                exit_price = df.iloc[exit_idx]['close']
                
                # حساب العائد
                gross_ret = (exit_price - entry) / entry
                net_ret = gross_ret - 0.0013  # عمولة + انزلاق (~0.13%)
                
                # تطبيق وقف الخسارة وجني الربح
                if net_ret < -stop_loss:
                    net_ret = -stop_loss
                elif net_ret > take_profit:
                    net_ret = take_profit
                
                profit = current_capital * net_ret
                current_capital += profit
                
                # تتبع الانخفاض
                if current_capital > peak:
                    peak = current_capital
                dd = (peak - current_capital) / peak
                if dd > max_dd:
                    max_dd = dd
                
                trades.append({'return': net_ret})
                i += hold_period
            else:
                i += 1
        
        # التحليل
        total_trades = len(trades)
        winners = sum(1 for t in trades if t['return'] > 0)
        win_rate = (winners / total_trades * 100) if total_trades > 0 else 0
        total_ret = ((current_capital - capital) / capital) * 100
        
        avg_win = np.mean([t['return'] for t in trades if t['return'] > 0]) if winners > 0 else 0
        avg_loss = np.mean([t['return'] for t in trades if t['return'] <= 0]) if (total_trades - winners) > 0 else 0
        pf = abs(avg_win * winners / (avg_loss * (total_trades - winners))) if (total_trades - winners) > 0 and avg_loss != 0 else float('inf')
        
        print("\n" + "="*50)
        print("💰 النتائج")
        print("="*50)
        print(f"💵 ${capital:,.0f} → ${current_capital:,.0f}")
        print(f"📈 العائد: {total_ret:.2f}%")
        print(f"🔢 الصفقات: {total_trades}")
        print(f"🎯 الفوز: {win_rate:.1f}%")
        print(f"📊 متوسط الربح: {avg_win*100:.2f}%")
        print(f"📉 متوسط الخسارة: {avg_loss*100:.2f}%")
        print(f"💹 عامل الربح: {pf:.2f}")
        print(f"📉 أقصى انخفاض: {max_dd*100:.2f}%")
        print("="*50)
        
        if win_rate >= 50 and total_ret >= 10:
            print("🌟 ممتاز!")
        elif win_rate >= 40 and total_ret >= 0:
            print("⚠️ جيد")
        else:
            print("❌ ضعيف")
        
        return current_capital, win_rate, total_trades

# === التشغيل ===
if __name__ == "__main__":
    trader = SimpleProfitableTrader(symbol='BTC/USDT', timeframe='4h')
    
    try:
        trader.fetch_data(days=730)
        trader.add_indicators()
        
        print("\n" + "="*50)
        print("🔬 اختبار معاملات مختلفة")
        print("="*50)
        
        best = None
        best_params = {}
        
        for hold in [3, 5, 7]:
            for sl in [0.02, 0.03]:
                for tp in [0.04, 0.06]:
                    print(f"\n📊 Hold={hold}, SL={sl*100:.0f}%, TP={tp*100:.0f}%")
                    cap, wr, trades = trader.run_backtest(hold_period=hold, stop_loss=sl, take_profit=tp)
                    
                    if best is None or (wr >= 45 and cap > best[0]):
                        best = (cap, wr, trades)
                        best_params = {'hold': hold, 'sl': sl, 'tp': tp}
        
        print("\n" + "="*50)
        print("🏆 أفضل نتيجة")
        print("="*50)
        print(f"📊 المعاملات المثلى:")
        print(f"   فترة الثبات: {best_params['hold']} شموع")
        print(f"   وقف الخسارة: {best_params['sl']*100:.0f}%")
        print(f"   جني الربح: {best_params['tp']*100:.0f}%")
        print(f"\n💵 النهائي: ${best[0]:,.0f}")
        print(f"🎯 الفوز: {best[1]:.1f}%")
        print(f"🔢 الصفقات: {best[2]}")
        
        print("\n✅ انتهى!")
        
    except Exception as e:
        print(f"❌ خطأ: {e}")
