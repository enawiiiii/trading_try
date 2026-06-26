import pandas as pd
import numpy as np
import ccxt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
import warnings
warnings.filterwarnings('ignore')

class ProfitableAI_Trader:
    """
    نظام تداول ذكي محسّن للربحية العالية
    - يركز على الجودة لا الكمية
    - فلتر اتجاه قوي جداً
    - إدارة مخاطر صارمة
    - أهداف ربح واقعية
    """
    
    def __init__(self, symbol='BTC/USDT', timeframe='4h', lookback_days=730):
        self.symbol = symbol
        self.timeframe = timeframe  # فريم زمني أكبر لتقليل الضوضاء
        self.lookback_days = lookback_days
        self.exchange = ccxt.binance()
        self.df = None
        self.model = None
        
        # إعدادات واقعية
        self.fee = 0.0004
        self.slippage = 0.0005
        
        print(f"🤖 تهيئة نظام التداول الربحي لـ {symbol}...")

    def fetch_data(self):
        """جلب بيانات حقيقية"""
        print("📡 جاري جلب البيانات من Binance...")
        bars = self.lookback_days * 6 if self.timeframe == '4h' else self.lookback_days * 24
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=bars)
        self.df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], unit='ms')
        self.df.set_index('timestamp', inplace=True)
        print(f"✅ تم جلب {len(self.df)} شمعة.")
        return self.df

    def engineer_features(self):
        """بناء مؤشرات مركزة على الجودة"""
        df = self.df.copy()
        
        # ====== الاتجاه القوي ======
        df['sma_50'] = SMAIndicator(df['close'], window=50).sma_indicator()
        df['sma_200'] = SMAIndicator(df['close'], window=200).sma_indicator()
        df['ema_20'] = EMAIndicator(df['close'], window=20).ema_indicator()
        df['ema_50'] = EMAIndicator(df['close'], window=50).ema_indicator()
        df['adx'] = ADXIndicator(df['high'], df['low'], df['close']).adx()
        
        # فلتر الاتجاه الصاعد القوي
        df['strong_uptrend'] = (
            (df['close'] > df['sma_200']) & 
            (df['close'] > df['sma_50']) & 
            (df['ema_20'] > df['ema_50']) &
            (df['adx'] > 20)
        ).astype(int)
        
        # ====== الزخم ======
        df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
        df['stoch_k'] = StochasticOscillator(df['high'], df['low'], df['close']).stoch()
        
        # RSI في منطقة صاعدة لكن غير متطرفة
        df['rsi_good'] = ((df['rsi'] > 45) & (df['rsi'] < 70)).astype(int)
        
        # ====== التقلب ======
        bb = BollingerBands(df['close'])
        df['bb_low'] = bb.bollinger_lband()
        df['bb_high'] = bb.bollinger_hband()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['sma_50']
        df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # السعر قريب من الحد السفلي لبولينجر (فرصة شراء جيدة)
        df['near_bb_low'] = ((df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low']) < 0.3).astype(int)
        
        # ====== الحجم ======
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # ====== MACD ======
        macd = MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # تقاطع إيجابي
        df['macd_bullish'] = ((df['macd'] > df['macd_signal']) & 
                              (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        
        # ====== العوائد ======
        df['return_1'] = df['close'].pct_change()
        df['return_3'] = df['close'].pct_change(3)
        df['return_5'] = df['close'].pct_change(5)
        
        # ====== الدرجة المركبة ======
        # دمج الإشارات الإيجابية
        df['composite_score'] = (
            df['strong_uptrend'] * 2 +
            df['rsi_good'] +
            df['near_bb_low'] +
            df['macd_bullish'] +
            (df['volume_ratio'] > 1.2).astype(int)
        )
        
        df.dropna(inplace=True)
        self.df = df
        print(f"🧠 تم بناء {len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']])} مؤشر.")
        return df

    def create_target(self, lookforward=5, min_return=0.02):
        """هدف واقعي: ربح 2% خلال 5 شموع"""
        df = self.df.copy()
        forward_return = df['close'].shift(-lookforward) / df['close'] - 1
        net_return = forward_return - (self.fee * 2) - self.slippage
        
        # هدف أعلى جودة: ربح صافي > 2%
        df['target'] = (net_return > min_return).astype(int)
        df['forward_return'] = forward_return
        
        self.df = df
        return df

    def train_model(self):
        """تدريب نموذج يركز على الدقة"""
        features = [col for col in self.df.columns 
                   if col not in ['target', 'open', 'high', 'low', 'close', 'volume', 'forward_return']]
        
        X = self.df[features]
        y = self.df['target']
        
        split_idx = int(len(X) * 0.75)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print("🏋️‍♂️ تدريب نموذج Random Forest المتقدم...")
        print(f"   - تدريب: {len(X_train)}, اختبار: {len(X_test)}")
        
        # Random Forest أكثر استقراراً مع البيانات المحدودة
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        print(f"\n📊 الدقة: {acc:.2%}")
        print("\n📈 التقرير التفصيلي:")
        print(classification_report(y_test, y_pred, target_names=['Wait', 'Buy']))
        
        # أهمية المؤشرات
        feature_imp = pd.DataFrame({
            'feature': features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔝 أهم 5 مؤشرات:")
        print(feature_imp.head(5).to_string(index=False))
        
        return X_test, y_test, y_pred, y_proba

    def run_backtest(self, X_test, y_test, predictions, probas, min_confidence=0.65):
        """باك تست محافظ يركز على الجودة"""
        test_df = self.df.loc[X_test.index].copy()
        test_df['pred'] = predictions
        test_df['proba'] = probas
        
        capital = 10000
        current_capital = capital
        trades = []
        peak = capital
        max_dd = 0
        
        print("\n🚀 بدء المحاكاة...")
        print(f"🎯 الحد الأدنى للثقة: {min_confidence:.0%}")
        
        i = 0
        while i < len(test_df) - 5:
            row = test_df.iloc[i]
            
            # شروط دخول صارمة جداً:
            # 1. تنبؤ بالشراء
            # 2. ثقة عالية
            # 3. اتجاه صاعد قوي
            if row['pred'] == 1 and row['proba'] >= min_confidence and row['strong_uptrend'] == 1:
                entry = row['close']
                exit_idx = i + 5
                
                if exit_idx >= len(test_df):
                    break
                
                exit_price = test_df.iloc[exit_idx]['close']
                
                gross_ret = (exit_price - entry) / entry
                net_ret = gross_ret - (self.fee * 2) - self.slippage
                
                # وقف خسارة 3%، هدف 6%
                if net_ret < -0.03:
                    net_ret = -0.03
                elif net_ret > 0.06:
                    net_ret = 0.06
                
                profit = current_capital * net_ret
                current_capital += profit
                
                if current_capital > peak:
                    peak = current_capital
                dd = (peak - current_capital) / peak
                if dd > max_dd:
                    max_dd = dd
                
                trades.append({'return': net_ret, 'proba': row['proba']})
                i += 5
            else:
                i += 1
        
        total_trades = len(trades)
        winners = sum(1 for t in trades if t['return'] > 0)
        win_rate = (winners / total_trades * 100) if total_trades > 0 else 0
        total_ret = ((current_capital - capital) / capital) * 100
        
        avg_win = np.mean([t['return'] for t in trades if t['return'] > 0]) if winners > 0 else 0
        avg_loss = np.mean([t['return'] for t in trades if t['return'] <= 0]) if (total_trades - winners) > 0 else 0
        pf = abs(avg_win * winners / (avg_loss * (total_trades - winners))) if (total_trades - winners) > 0 and avg_loss != 0 else float('inf')
        
        print("\n" + "="*50)
        print("💰 النتائج النهائية")
        print("="*50)
        print(f"💵 رأس المال: ${capital:,.0f} → ${current_capital:,.0f}")
        print(f"📈 العائد: {total_ret:.2f}%")
        print(f"🔢 الصفقات: {total_trades}")
        print(f"🎯 نسبة الفوز: {win_rate:.2f}%")
        print(f"📊 متوسط الربح: {avg_win*100:.2f}%")
        print(f"📉 متوسط الخسارة: {avg_loss*100:.2f}%")
        print(f"💹 عامل الربح: {pf:.2f}")
        print(f"📉 أقصى انخفاض: {max_dd*100:.2f}%")
        print("="*50)
        
        if win_rate >= 55 and total_ret >= 15 and pf > 1.5:
            print("🌟 ممتاز! جاهز للتجربة الديمو")
        elif win_rate >= 45 and total_ret >= 0:
            print("⚠️ جيد - يحتاج تحسينات")
        else:
            print("❌ ضعيف - يحتاج إعادة تصميم")
        
        return current_capital, win_rate, total_trades, max_dd, pf

# === التشغيل ===
if __name__ == "__main__":
    trader = ProfitableAI_Trader(symbol='BTC/USDT', timeframe='4h', lookback_days=730)
    
    try:
        trader.fetch_data()
        trader.engineer_features()
        trader.create_target(lookforward=5, min_return=0.02)
        X_test, y_test, preds, probas = trader.train_model()
        
        print("\n" + "="*50)
        print("🔬 اختبار عتبات مختلفة")
        print("="*50)
        
        best = None
        best_thresh = 0.65
        
        for thresh in [0.60, 0.65, 0.70, 0.75]:
            print(f"\n{'='*50}")
            print(f"🎯 عتبة {thresh:.0%}")
            print("="*50)
            cap, wr, trades, dd, pf = trader.run_backtest(X_test, y_test, preds, probas, min_confidence=thresh)
            
            if best is None or (wr >= 50 and cap > best[0]):
                best = (cap, wr, trades, dd, pf)
                best_thresh = thresh
        
        print("\n" + "="*50)
        print("🏆 أفضل نتيجة")
        print("="*50)
        print(f"🎯 العتبة المثلى: {best_thresh:.0%}")
        print(f"💵 النهائي: ${best[0]:,.0f}")
        print(f"🎯 الفوز: {best[1]:.1f}%")
        print(f"🔢 الصفقات: {best[2]}")
        print(f"📉 الانخفاض: {best[3]*100:.1f}%")
        print(f"💹 العامل: {best[4]:.2f}")
        
        print("\n✅ انتهى التحليل!")
        
    except Exception as e:
        print(f"❌ خطأ: {e}")
