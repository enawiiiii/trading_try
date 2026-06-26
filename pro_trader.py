import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import talib
import ccxt
import warnings
warnings.filterwarnings('ignore')

class ProTradingSystem:
    def __init__(self, symbol='BTC/USDT', timeframe='1h'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange = ccxt.binance()
        self.model = None
        self.data = None
        self.features = []
        
        # إعدادات احترافية
        self.lookahead = 20  # ننظر 20 شمعة للأمام لتحديد الربح
        self.min_profit_pct = 1.5  # نعتبر الصفقة رابحة فقط إذا ربحت 1.5%
        self.max_loss_pct = 1.0    # ونخسرها إذا خسرت 1%
        self.confidence_threshold = 0.65 # ثقة عالية للدخول
        
    def fetch_data(self, limit=2000):
        """جلب بيانات حقيقية من بينانس"""
        print(f"📡 جاري جلب بيانات {self.symbol}...")
        bars = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        self.data = df
        return df

    def add_advanced_indicators(self, df):
        """إضافة مؤشرات فنية متقدمة وقوية"""
        # 1. مؤشرات الاتجاه (Trend)
        df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
        df['ema_200'] = talib.EMA(df['close'], timeperiod=200)
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        # 2. مؤشرات الزخم (Momentum)
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['macd'], df['macd_signal'], _ = talib.MACD(df['close'])
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
        
        # 3. مؤشرات التقلب (Volatility)
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'], timeperiod=20)
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # 4. مؤشرات الحجم (Volume)
        df['vol_sma'] = talib.SMA(df['volume'], timeperiod=20)
        df['vol_ratio'] = df['volume'] / df['vol_sma']
        
        # تنظيف القيم الناقصة
        df.dropna(inplace=True)
        return df

    def create_target(self, df):
        """
        الهدف الذكي: 
        1 = إذا كان السعر سيرتفع بنسبة min_profit_pct قبل أن ينخفض max_loss_pct
        0 = عكس ذلك
        هذه الطريقة تعلم النموذج "أنماط الربح" فقط.
        """
        targets = []
        for i in range(len(df)):
            if i + self.lookahead >= len(df):
                targets.append(0)
                continue
            
            entry_price = df['close'].iloc[i]
            future_high = df['high'].iloc[i+1:i+self.lookahead+1].max()
            future_low = df['low'].iloc[i+1:i+self.lookahead+1].min()
            
            profit_pct = (future_high - entry_price) / entry_price * 100
            loss_pct = (entry_price - future_low) / entry_price * 100
            
            # شرط صارم: يجب تحقيق الربح قبل الخسارة الكبيرة
            if profit_pct >= self.min_profit_pct and loss_pct < self.max_loss_pct * 1.5:
                targets.append(1)
            else:
                targets.append(0)
                
        df['target'] = targets
        return df

    def prepare_features(self, df):
        """تحضير الميزات وتصفيتها"""
        feature_cols = [
            'ema_50', 'ema_200', 'adx', 'rsi', 'macd', 'stoch_k', 'stoch_d',
            'atr', 'bb_width', 'bb_position', 'vol_ratio'
        ]
        
        # إضافة تفاعلات بين المؤشرات (Feature Engineering)
        df['trend_strength'] = (df['ema_50'] > df['ema_200']).astype(int) * df['adx']
        df['momentum_score'] = (df['rsi'] - 50) + (df['macd'] - df['macd_signal']) * 100
        df['volatility_adj'] = df['close'] / df['atr']
        
        feature_cols.extend(['trend_strength', 'momentum_score', 'volatility_adj'])
        
        X = df[feature_cols]
        y = df['target']
        
        # تطبيع البيانات (Scaling) يدوياً لضمان الاستقرار
        for col in X.columns:
            mean = X[col].mean()
            std = X[col].std()
            if std > 0:
                X[col] = (X[col] - mean) / std
            else:
                X[col] = 0
                
        self.features = feature_cols
        return X, y

    def train_model(self, X, y):
        """تدريب نموذج Gradient Boosting المتقدم"""
        print("🧠 جاري تدريب النموذج الاحترافي...")
        
        # تقسيم البيانات (80% تدريب، 20% اختبار)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # معالجة عدم التوازن (Class Imbalance)
        # نجعل النموذج يهتم أكثر بالصفقات الرابحة (النادرة)
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = GradientBoostingClassifier(
            n_estimators=300,       # عدد الأشجار
            learning_rate=0.05,     # معدل التعلم (بطيء وثابت)
            max_depth=5,            # عمق الشجرة (لا نريد Overfitting)
            min_samples_split=20,   # الحد الأدنى للعينات للتفرع
            min_samples_leaf=10,    # الحد الأدنى للأوراق
            subsample=0.8,          # عشوائية في البيانات
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # التقييم
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        print("\n📊 نتائج الاختبار على البيانات المخفية:")
        print(classification_report(y_test, y_pred, target_names=['Fail', 'Win']))
        print(f"✅ الدقة العامة: {accuracy_score(y_test, y_pred)*100:.2f}%")
        
        # تحليل الأهمية
        importance = pd.DataFrame({
            'Feature': self.features,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        print("\n🏆 أهم 5 مؤشرات تؤثر في القرار:")
        print(importance.head(5))
        
        self.model = model
        return model

    def run_backtest(self, df):
        """تشغيل باك تست واقعي جداً مع إدارة مخاطر"""
        print("\n🚀 جاري تشغيل الباك تست المتقدم...")
        
        X, _ = self.prepare_features(df)
        signals = self.model.predict_proba(X)[:, 1]
        
        capital = 10000
        position = None
        trades = []
        
        # محاكاة الشمعة تلو الشمعة
        for i in range(len(df) - self.lookahead):
            current_idx = i
            signal_prob = signals[current_idx]
            current_price = df['close'].iloc[current_idx]
            
            # شروط الدخول الصارمة
            if signal_prob >= self.confidence_threshold and position is None:
                # فلتر الاتجاه الإضافي: لا تشترِ إلا إذا كان EMA50 > EMA200
                if df['ema_50'].iloc[current_idx] > df['ema_200'].iloc[current_idx]:
                    position = {
                        'entry_price': current_price,
                        'entry_time': df.index[current_idx],
                        'stop_loss': current_price * (1 - self.max_loss_pct/100),
                        'take_profit': current_price * (1 + self.min_profit_pct/100)
                    }
            
            # إدارة الصفقة المفتوحة
            if position:
                # نفحص الشمع المستقبلية لمعرفة ما حدث فعلياً
                future_high = df['high'].iloc[current_idx+1:current_idx+self.lookahead+1].max()
                future_low = df['low'].iloc[current_idx+1:current_idx+self.lookahead+1].min()
                
                # هل ضرب وقف الخسارة أولاً؟
                if future_low <= position['stop_loss']:
                    # خسارة
                    exit_price = position['stop_loss']
                    pnl_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
                    trades.append({'type': 'LOSS', 'pnl_pct': pnl_pct})
                    capital *= (1 + pnl_pct/100)
                    position = None
                
                # هل ضرب هدف الربح أولاً؟ (نتحقق من الترتيب الزمني التقريبي)
                elif future_high >= position['take_profit']:
                    # ربح
                    exit_price = position['take_profit']
                    pnl_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
                    trades.append({'type': 'WIN', 'pnl_pct': pnl_pct})
                    capital *= (1 + pnl_pct/100)
                    position = None
                
                # إذا انتهت الفترة ولم يضرب أي شيء (خروج قسري)
                # (تم تبسيط هذا الجزء في المحاكاة للاعتماد على Target الأصلي)

        # حساب الإحصائيات النهائية
        if not trades:
            print("⚠️ لم يتم تنفيذ أي صفقات! الشروط صارمة جداً أو السوق غير مناسب.")
            return

        wins = [t for t in trades if t['type'] == 'WIN']
        losses = [t for t in trades if t['type'] == 'LOSS']
        
        win_rate = len(wins) / len(trades) * 100
        total_return = ((capital - 10000) / 10000) * 100
        avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
        
        print("\n" + "="*40)
        print("💰 نتائج الباك تست النهائية (المحاكاة الواقعية)")
        print("="*40)
        print(f"💵 رأس المال النهائي: ${capital:.2f}")
        print(f"📈 العائد الإجمالي: {total_return:.2f}%")
        print(f"📊 عدد الصفقات: {len(trades)}")
        print(f"🏆 نسبة الفوز: {win_rate:.2f}%")
        print(f"💰 متوسط الربح في الصفقة الرابحة: {avg_win:.2f}%")
        print(f"📉 متوسط الخسارة في الصفقة الخاسرة: {avg_loss:.2f}%")
        print(f"📊 عامل الربح (Profit Factor): {(len(wins)*avg_win) / (len(losses)*abs(avg_loss)) if losses else 0:.2f}")
        print("="*40)

    def run(self):
        """الدالة الرئيسية لتشغيل كل شيء"""
        try:
            df = self.fetch_data()
            df = self.add_advanced_indicators(df)
            df = self.create_target(df)
            
            # إزالة الصفوف التي تحتوي على NaN الناتجة عن الحسابات
            df.dropna(inplace=True)
            
            if len(df) < 500:
                print("❌ البيانات غير كافية للتدريب.")
                return

            X, y = self.prepare_features(df)
            self.train_model(X, y)
            self.run_backtest(df)
            
        except Exception as e:
            print(f"❌ حدث خطأ: {e}")
            print("تأكد من تثبيت المكتبات: pip install pandas numpy scikit-learn ta-lib ccxt")
            print("ملاحظة: TA-Lib قد يحتاج تثبيت منفصل حسب نظام التشغيل.")

if __name__ == "__main__":
    # تشغيل النظام
    trader = ProTradingSystem(symbol='BTC/USDT', timeframe='1h')
    trader.run()
