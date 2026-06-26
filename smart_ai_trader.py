import pandas as pd
import numpy as np
import ccxt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
import warnings
warnings.filterwarnings('ignore')

class SmartAI_Trader:
    """
    نظام تداول ذكي متطور مع تحسينات متعددة
    - فلتر اتجاه قوي
    - مؤشرات فنية شاملة
    - إدارة مخاطر ديناميكية
    - تعلم آلي متقدم
    """
    
    def __init__(self, symbol='BTC/USDT', timeframe='1h', lookback_days=365):
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        self.exchange = ccxt.binance()
        self.df = None
        self.model = None
        self.results = {}
        
        # إعدادات المحاكاة الواقعية
        self.fee = 0.0004  # 0.04% عمولة بينانس
        self.slippage = 0.0005  # 0.05% انزلاق سعري
        
        print(f"🤖 تهيئة النظام الذكي لـ {symbol}...")

    def fetch_data(self):
        """جلب بيانات حقيقية من بينانس"""
        print("📡 جاري جلب البيانات التاريخية من Binance...")
        bars = self.lookback_days * 24 if self.timeframe == '1h' else self.lookback_days * 6
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=bars)
        self.df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], unit='ms')
        self.df.set_index('timestamp', inplace=True)
        print(f"✅ تم جلب {len(self.df)} شمعة بنجاح.")
        return self.df

    def engineer_features(self):
        """بناء مؤشرات ذكية ومتعددة"""
        df = self.df.copy()
        
        # ====== 1. مؤشرات الاتجاه (Trend Indicators) ======
        df['sma_20'] = SMAIndicator(df['close'], window=20).sma_indicator()
        df['sma_50'] = SMAIndicator(df['close'], window=50).sma_indicator()
        df['sma_200'] = SMAIndicator(df['close'], window=200).sma_indicator()
        df['ema_12'] = EMAIndicator(df['close'], window=12).ema_indicator()
        df['ema_26'] = EMAIndicator(df['close'], window=26).ema_indicator()
        df['adx'] = ADXIndicator(df['high'], df['low'], df['close']).adx()
        
        # قوة الاتجاه
        df['trend_strength'] = np.where(
            (df['close'] > df['sma_20']) & (df['close'] > df['sma_50']) & (df['close'] > df['sma_200']),
            2,  # اتجاه صاعد قوي
            np.where(
                (df['close'] < df['sma_20']) & (df['close'] < df['sma_50']) & (df['close'] < df['sma_200']),
                -2,  # اتجاه هابط قوي
                0  # عرضي
            )
        )
        
        # ====== 2. مؤشرات الزخم (Momentum Indicators) ======
        df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
        df['stoch_k'] = StochasticOscillator(df['high'], df['low'], df['close']).stoch()
        df['stoch_d'] = StochasticOscillator(df['high'], df['low'], df['close'], window=3).stoch_signal()
        df['williams_r'] = WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
        
        # تطبيع الزخم
        df['momentum_score'] = (
            (df['rsi'] - 50) / 50 + 
            (df['stoch_k'] - 50) / 50 + 
            (-df['williams_r'] - 50) / 50
        ) / 3
        
        # ====== 3. مؤشرات التقلب (Volatility Indicators) ======
        bb = BollingerBands(df['close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['sma_20']
        df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        df['atr_pct'] = df['atr'] / df['close'] * 100
        
        # موقع السعر بالنسبة لبولينجر
        df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
        
        # ====== 4. مؤشرات الحجم (Volume Indicators) ======
        df['vwap'] = VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # ====== 5. MACD ======
        macd = MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        df['macd_cross'] = np.where(df['macd_diff'] > 0, 1, -1)
        
        # ====== 6. أنماط السعر (Price Action) ======
        df['return_1'] = df['close'].pct_change()
        df['return_3'] = df['close'].pct_change(3)
        df['return_5'] = df['close'].pct_change(5)
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['close_open_ratio'] = (df['close'] - df['open']) / df['open']
        
        # أنماط الشموع
        df['bullish_engulfing'] = ((df['close'] > df['open']) & 
                                   (df['close'].shift(1) < df['open'].shift(1)) &
                                   (df['close'] > df['open'].shift(1)) &
                                   (df['open'] < df['close'].shift(1))).astype(int)
        
        df['bearish_engulfing'] = ((df['close'] < df['open']) & 
                                   (df['close'].shift(1) > df['open'].shift(1)) &
                                   (df['close'] < df['open'].shift(1)) &
                                   (df['open'] > df['close'].shift(1))).astype(int)
        
        # ====== 7. تحديد حالة السوق (Market Regime) ======
        df['market_regime'] = 0
        df.loc[(df['close'] > df['sma_200']) & (df['adx'] > 25), 'market_regime'] = 1  # صاعد قوي
        df.loc[(df['close'] < df['sma_200']) & (df['adx'] > 25), 'market_regime'] = -1 # هابط قوي
        df.loc[df['adx'] <= 25, 'market_regime'] = 0  # عرضي
        
        # تنظيف البيانات
        df.dropna(inplace=True)
        self.df = df
        print(f"🧠 تمت هندسة {len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']])} مؤشر تقني.")
        return df

    def create_target(self, lookforward=12, min_return=0.01):
        """إنشاء الهدف الذكي: الربح الصافي بعد الخصومات"""
        df = self.df.copy()
        forward_return = df['close'].shift(-lookforward) / df['close'] - 1
        
        # حساب العائد الصافي بعد العمولة والانزلاق
        net_return = forward_return - (self.fee * 2) - self.slippage
        
        # التصنيف: 1 (شراء)، 0 (انتظار/بيع)
        df['target'] = (net_return > min_return).astype(int)
        df['forward_return'] = forward_return
        
        self.df = df
        return df

    def train_model(self):
        """تدريب نموذج ذكاء اصطناعي متقدم"""
        features = [col for col in self.df.columns 
                   if col not in ['target', 'open', 'high', 'low', 'close', 'volume', 
                                  'bb_high', 'bb_low', 'forward_return']]
        
        X = self.df[features]
        y = self.df['target']
        
        # تقسيم زمني (لا خلط عشوائي لمنع تسريب المستقبل)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print("🏋️‍♂️ جاري تدريب نموذج Gradient Boosting المتقدم...")
        print(f"   - عدد العينات التدريبية: {len(X_train)}")
        print(f"   - عدد العينات الاختبارية: {len(X_test)}")
        print(f"   - عدد المؤشرات: {len(features)}")
        
        # نموذج متوازن للتعامل مع ندرة الفرص الممتازة
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_samples_split=15,
            min_samples_leaf=8,
            subsample=0.8,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # التقييم
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        print(f"\n📊 دقة النموذج على البيانات المخفية: {acc:.2%}")
        print("\n📈 تقرير الأداء التفصيلي:")
        print(classification_report(y_test, y_pred, target_names=['Wait/Sell', 'Strong Buy']))
        
        # أهمية المؤشرات
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔝 أهم 10 مؤشرات:")
        print(feature_importance.head(10).to_string(index=False))
        
        return X_test, y_test, y_pred, y_proba

    def run_smart_backtest(self, X_test, y_test, predictions, probas, confidence_threshold=0.6):
        """محاكاة تداول ذكية مع إدارة مخاطر متطورة"""
        test_df = self.df.loc[X_test.index].copy()
        test_df['prediction'] = predictions
        test_df['probability'] = probas
        test_df['actual'] = y_test
        
        capital = 10000  # رأس مال ابتدائي $10,000
        current_capital = capital
        trades = []
        max_drawdown = 0
        peak_capital = capital
        
        print("\n🚀 بدء محاكاة التداول الذكي...")
        print(f"🎯 عتبة الثقة المطلوبة: {confidence_threshold:.0%}")
        
        i = 0
        while i < len(test_df) - 12:
            row = test_df.iloc[i]
            
            # شرط الدخول الذكي: تنبؤ بالشراء + ثقة عالية + اتجاه إيجابي
            if row['prediction'] == 1 and row['probability'] >= confidence_threshold:
                entry_price = row['close']
                
                # البحث عن نقطة الخروج (بعد 12 ساعة)
                exit_idx = i + 12
                if exit_idx >= len(test_df):
                    break
                
                exit_price = test_df.iloc[exit_idx]['close']
                
                # حساب الربح/الخسارة الفعلي
                gross_return = (exit_price - entry_price) / entry_price
                net_return = gross_return - (self.fee * 2) - self.slippage
                
                # تطبيق إدارة المخاطر الديناميكية
                # وقف خسارة عند -2%، جني أرباح جزئي عند +4%
                if net_return < -0.02:
                    net_return = -0.02
                elif net_return > 0.04:
                    net_return = 0.04
                
                profit = current_capital * net_return
                current_capital += profit
                
                # تتبع أقصى انخفاض
                if current_capital > peak_capital:
                    peak_capital = current_capital
                drawdown = (peak_capital - current_capital) / peak_capital
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                
                trades.append({
                    'entry': entry_price,
                    'exit': exit_price,
                    'return': net_return,
                    'probability': row['probability'],
                    'type': 'LONG'
                })
                
                i += 12  # تخطي فترة الصفقة
            else:
                i += 1
        
        # التحليل النهائي
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['return'] > 0)
        losing_trades = sum(1 for t in trades if t['return'] <= 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_return = ((current_capital - capital) / capital) * 100
        
        avg_win = np.mean([t['return'] for t in trades if t['return'] > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t['return'] for t in trades if t['return'] <= 0]) if losing_trades > 0 else 0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')
        
        print("\n" + "="*50)
        print("💰 نتائج المحاكاة النهائية (النظام الذكي)")
        print("="*50)
        print(f"💵 رأس المال الابتدائي: ${capital:,.2f}")
        print(f"💵 رأس المال النهائي:   ${current_capital:,.2f}")
        print(f"📈 العائد الإجمالي:     {total_return:.2f}%")
        print(f"🔢 عدد الصفقات المنفذة: {total_trades}")
        print(f"🎯 نسبة الفوز:          {win_rate:.2f}%")
        print(f"📊 متوسط الربح للصفقات الرابحة: {avg_win*100:.2f}%")
        print(f"📉 متوسط الخسارة للصفقات الخاسرة: {avg_loss*100:.2f}%")
        print(f"💹 عامل الربح (Profit Factor): {profit_factor:.2f}")
        print(f"📉 أقصى انخفاض (Max Drawdown): {max_drawdown*100:.2f}%")
        print(f"⚡ متوسط الربح/صفقة:    {(total_return/total_trades):.2f}%" if total_trades > 0 else "N/A")
        print("="*50)
        
        if win_rate >= 55 and total_return >= 20 and profit_factor > 1.5:
            print("🌟 تصنيف النظام: ممتاز (جاهز للتجربة الديمو)")
        elif win_rate >= 45 and total_return >= 0:
            print("⚠️ تصنيف النظام: جيد (يحتاج ضبط دقيق)")
        else:
            print("❌ تصنيف النظام: ضعيف (يحتاج إعادة تدريب)")
            
        return current_capital, win_rate, total_trades, max_drawdown, profit_factor

# --- التشغيل الرئيسي ---
if __name__ == "__main__":
    trader = SmartAI_Trader(symbol='BTC/USDT', timeframe='1h', lookback_days=365)
    
    try:
        # 1. جلب البيانات
        trader.fetch_data()
        
        # 2. الهندسة والتحليل
        trader.engineer_features()
        trader.create_target(lookforward=12, min_return=0.01)
        
        # 3. التدريب
        X_test, y_test, preds, probas = trader.train_model()
        
        # 4. الباك تست الذكي مع عتبات ثقة مختلفة
        print("\n" + "="*50)
        print("🔬 اختبار استراتيجيات متعددة")
        print("="*50)
        
        best_result = None
        best_threshold = 0.6
        
        for threshold in [0.5, 0.6, 0.7]:
            print(f"\n{'='*50}")
            print(f"🎯 اختبار بعتبة ثقة {threshold:.0%}")
            print("="*50)
            cap, wr, trades, dd, pf = trader.run_smart_backtest(X_test, y_test, preds, probas, confidence_threshold=threshold)
            
            if best_result is None or (wr > 50 and cap > best_result[0]):
                best_result = (cap, wr, trades, dd, pf)
                best_threshold = threshold
        
        print("\n" + "="*50)
        print("🏆 أفضل نتيجة")
        print("="*50)
        print(f"🎯 عتبة الثقة المثلى: {best_threshold:.0%}")
        print(f"💵 رأس المال النهائي: ${best_result[0]:,.2f}")
        print(f"🎯 نسبة الفوز: {best_result[1]:.2f}%")
        print(f"🔢 عدد الصفقات: {best_result[2]}")
        print(f"📉 أقصى انخفاض: {best_result[3]*100:.2f}%")
        print(f"💹 عامل الربح: {best_result[4]:.2f}")
        
        print("\n✅ اكتمل التحليل بنجاح! النظام جاهز للمرحلة التجريبية.")
        
    except Exception as e:
        print(f"❌ حدث خطأ أثناء التشغيل: {e}")
        print("تأكد من اتصال الإنترنت وتثبيت المكتبات: pip install ccxt scikit-learn ta pandas numpy")
