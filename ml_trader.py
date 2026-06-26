import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, classification_report
# talib تم استبداله بحسابات يدوية لتجنب مشاكل التثبيت

class ProfessionalMLTrader:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,  # زيادة عدد الأشجار لدقة أعلى
            max_depth=15,      # عمق أكبر للتعلم
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1          # استخدام كل الأنوية للمعالجة الأسرع
        )
        self.is_trained = False
        self.feature_columns = []

    def prepare_features(self, df):
        """
        هندسة سمات متقدمة: لا نستخدم السعر الخام، بل العلاقات الرياضية
        """
        df = df.copy()
        
        # 1. مؤشرات الاتجاه (Trend)
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['price_vs_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
        
        # 2. MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 3. مؤشرات الزخم (Momentum)
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        df['rsi_7'] = self._calculate_rsi(df['close'], 7)
        df['mom_10'] = df['close'].pct_change(periods=10)
        df['mom_5'] = df['close'].pct_change(periods=5)
        
        # 4. التقلب (Volatility)
        df['volatility'] = df['close'].rolling(window=14).std() / df['close'].rolling(window=14).mean()
        df['atr'] = self._calculate_atr(df)
        
        # 5. حجم التداول النسبي
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        # 6. نطاق السعر (High-Low Range)
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # 7. Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # تنظيف القيم_nan الناتجة عن الحسابات
        df.dropna(inplace=True)
        
        return df

    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr(self, df, period=14):
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def create_target(self, df, future_period=5, threshold=0.02):
        """
        الهدف: هل السعر سيرتفع بأكثر من 2% خلال الـ 5 شموع القادمة؟
        1 = نعم (شراء)، 0 = لا (انتظار/بيع)
        رفعنا العتبة إلى 2% لتجنب الإشارات الضعيفة
        """
        future_price = df['close'].shift(-future_period)
        returns = (future_price - df['close']) / df['close']
        target = (returns > threshold).astype(int)
        return target

    def train(self, historical_data):
        """
        تدريب النموذج على البيانات التاريخية
        """
        print("🔄 جاري تحضير البيانات وهندسة السمات...")
        df = self.prepare_features(historical_data)
        
        if len(df) < 100:
            print("❌ البيانات غير كافية للتدريب (أقل من 100 شمعة)")
            return False

        # إنشاء الهدف (Target)
        df['target'] = self.create_target(df)
        df.dropna(inplace=True) # حذف الصفوف الأخيرة التي لا تحتوي على مستقبل
        
        if df['target'].sum() == 0 or df['target'].sum() == len(df):
            print("❌ البيانات غير متوازنة تماماً، لا يمكن التدريب.")
            return False

        X = df.drop('target', axis=1)
        y = df['target']
        
        # إزالة عمود التاريخ إذا كان موجوداً لأنه لا يمكن استخدامه في التدريب
        if 'date' in X.columns:
            X = X.drop('date', axis=1)
        
        self.feature_columns = X.columns.tolist()
        
        # تقسيم البيانات (80% تدريب، 20% اختبار)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        print("🧠 جاري تدريب نموذج الغابة العشوائية (Random Forest)...")
        self.model.fit(X_train, y_train)
        
        # تقييم الأداء
        y_pred = self.model.predict(X_test)
        precision = precision_score(y_test, y_pred, zero_division=0)
        accuracy = (y_pred == y_test).mean()
        
        print(f"✅ اكتمل التدريب!")
        print(f"📊 دقة النموذج على بيانات الاختبار (Precision): {precision:.2f}")
        print(f"📈 نسبة الصحة العامة (Accuracy): {accuracy:.2f}")
        print("   (Precision تعني أنه عندما يتوقع صعوداً، تكون النسبة الصحيحة هي {})".format(precision))
        
        if precision < 0.5:
            print("⚠️ تحذير: دقة النموذج منخفضة، قد يحتاج إلى مزيد من البيانات أو ضبط المعاملات.")
        elif precision > 0.7:
            print("🎉 ممتاز! النموذج يظهر دقة عالية في التنبؤ.")
        
        self.is_trained = True
        return True

    def predict_signal(self, current_market_data):
        """
        اتخاذ قرار لحظي بناءً على آخر بيانات سوقية
        """
        if not self.is_trained:
            return "WAIT", 0.0

        # تحويل البيانات الحالية لتنسيق مناسب
        # نفترض أن current_market_data هو DataFrame يحتوي على آخر N شمعة لحساب المؤشرات
        df = self.prepare_features(current_market_data)
        
        if len(df) == 0:
            return "WAIT", 0.0
            
        last_row = df.iloc[[-1]] # آخر شمعة مكتملة
        
        # التأكد من تطابق الأعمدة
        try:
            last_row = last_row[self.feature_columns]
        except KeyError:
            return "WAIT", 0.0

        # التنبؤ والاحتمالية
        prediction = self.model.predict(last_row)[0]
        probability = self.model.predict_proba(last_row)[0][1] # احتمال الفئة 1 (صعود)
        
        decision = "BUY" if prediction == 1 and probability > 0.65 else "WAIT"
        # ملاحظة: رفعنا عتبة الاحتمال إلى 0.65 لضمان جودة الصفقات فقط
        
        return decision, probability

# --- مثال على كيفية الاستخدام والتجربة (Backtest Simulation) ---
if __name__ == "__main__":
    # 1. توليد بيانات وهمية للتجربة (في الواقع ستجلبها من API بينانس أو غيرها)
    # نستخدم freq='h' بدلاً من 'H' لتجنب التحذير
    dates = pd.date_range(start="2023-01-01", periods=2000, freq='h')
    np.random.seed(42)
    
    # محاكاة سعر أكثر واقعية مع اتجاهات واضحة
    base_price = 100
    returns = np.random.randn(2000) * 0.02 + 0.0005  # عائد يومي متوسط إيجابي بسيط
    close_prices = base_price * np.cumprod(1 + returns)
    
    # إضافة تقلبات يومية
    high_prices = close_prices * (1 + np.abs(np.random.randn(2000)) * 0.01)
    low_prices = close_prices * (1 - np.abs(np.random.randn(2000)) * 0.01)
    volume = np.random.randint(1000, 5000, 2000) * (close_prices / base_price)
    
    data = pd.DataFrame({
        'date': dates,
        'open': close_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })

    # 2. تشغيل المتداول المحترف
    trader = ProfessionalMLTrader()
    
    # تدريب
    success = trader.train(data)
    
    if success:
        print("\n--- بدء محاكاة التداول اللحظي ---")
        # محاكاة اتخاذ قرار على آخر 100 شمعة
        recent_data = data.tail(100)
        signal, prob = trader.predict_signal(recent_data)
        
        print(f"📡 الإشارة الحالية: {signal}")
        print(f"🎯 درجة الثقة: {prob:.2%}")
        
        if signal == "BUY":
            print("💰 تم تنفيذ أمر شراء افتراضي!")
            print("   - وقف الخسارة المقترح: 1.0%")
            print("   - هدف الربح المقترح: 2.5%")
        else:
            print("💤 الانتظار لفرصة أفضل...")
            print("   النظام يبحث عن فرص عالية الجودة فقط")
