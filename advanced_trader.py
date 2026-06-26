import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

class AdvancedMLTrader:
    """
    نظام تداول متقدم يستخدم تقنيات تعلم آلي متطورة
    مع إدارة مخاطر ديناميكية وفلترة ذكية للإشارات
    """
    
    def __init__(self, confidence_threshold=0.75):
        # استخدام Gradient Boosting بدلاً من Random Forest لدقة أعلى
        self.model = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = []
        self.confidence_threshold = confidence_threshold  # عتبة ثقة عالية للصفقات فقط
        
    def prepare_features(self, df):
        """
        هندسة سمات متقدمة جداً مع مؤشرات احترافية
        """
        df = df.copy()
        
        # === 1. مؤشرات الاتجاه المتعددة ===
        for period in [9, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # السعر بالنسبة للمتوسطات
        df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['price_vs_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
        df['price_vs_sma200'] = (df['close'] - df['sma_200']) / df['sma_200']
        
        # تقاطع المتوسطات
        df['sma20_above_sma50'] = (df['sma_20'] > df['sma_50']).astype(int)
        df['sma50_above_sma200'] = (df['sma_50'] > df['sma_200']).astype(int)
        
        # === 2. MACD المتقدم ===
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_cross_up'] = (df['macd'] > df['macd_signal']).astype(int)
        
        # === 3. RSI متعدد الفترات ===
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = self._calculate_rsi(df['close'], period)
        
        df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
        df['rsi_neutral'] = ((df['rsi_14'] >= 30) & (df['rsi_14'] <= 70)).astype(int)
        
        # === 4. الزخم والتسارع ===
        for period in [3, 5, 10, 20]:
            df[f'momentum_{period}'] = df['close'].pct_change(periods=period)
        
        df['acceleration'] = df['momentum_5'] - df['momentum_10']
        
        # === 5. التقلب الحقيقي (ATR) ===
        df['atr'] = self._calculate_atr(df, 14)
        df['atr_ratio'] = df['atr'] / df['close']
        df['volatility_expansion'] = df['atr_ratio'] > df['atr_ratio'].rolling(20).mean()
        
        # === 6. حجم التداول الذكي ===
        df['volume_sma20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma20']
        df['volume_spike'] = (df['volume_ratio'] > 1.5).astype(int)
        df['obv'] = self._calculate_obv(df)
        
        # === 7. Bollinger Bands المتقدم ===
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_squeeze'] = (df['bb_std'] < df['bb_std'].rolling(20).mean() * 0.8).astype(int)
        
        # === 8. أنماط الشموع ===
        df['body_size'] = abs(df['close'] - df['open']) / df['close']
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
        df['bullish_engulfing'] = ((df['close'] > df['open']) & 
                                   (df['close'].shift(1) < df['open'].shift(1)) &
                                   (df['close'] > df['open'].shift(1)) &
                                   (df['open'] < df['close'].shift(1))).astype(int)
        
        # === 9. اتجاه السوق العام ===
        df['trend_score'] = (
            (df['close'] > df['sma_20']).astype(int) +
            (df['close'] > df['sma_50']).astype(int) +
            (df['close'] > df['sma_200']).astype(int) +
            (df['sma_20'] > df['sma_50']).astype(int) +
            (df['sma_50'] > df['sma_200']).astype(int)
        ) / 5
        
        # تنظيف البيانات
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
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
    
    def _calculate_obv(self, df):
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv
    
    def create_target(self, df, future_period=5, threshold=0.025):
        """
        هدف أكثر صرامة: ربح 2.5% خلال 5 فترات
        مع فلتر لتجنب الأسواق الجانبية
        """
        future_price = df['close'].shift(-future_period)
        returns = (future_price - df['close']) / df['close']
        
        # شرط إضافي: تجنب الإشارات في الأسواق شديدة التقلب سلباً
        max_drawdown = df['close'].rolling(window=future_period).min().shift(-future_period)
        drawdown = (max_drawdown - df['close']) / df['close']
        
        target = ((returns > threshold) & (drawdown > -0.05)).astype(int)
        return target
    
    def train(self, historical_data, verbose=True):
        """
        تدريب متقدم مع توازن دقيق وتقييم شامل
        """
        if verbose:
            print("🔄 جاري تحضير البيانات وهندسة السمات المتقدمة...")
        
        df = self.prepare_features(historical_data)
        
        if len(df) < 200:
            print("❌ البيانات غير كافية (أقل من 200 شمعة)")
            return False
        
        # إنشاء الهدف
        df['target'] = self.create_target(df)
        df.dropna(inplace=True)
        
        # التحقق من توازن البيانات
        positive_ratio = df['target'].mean()
        if positive_ratio < 0.1 or positive_ratio > 0.9:
            print(f"⚠️ تحذير: البيانات غير متوازنة ({positive_ratio:.2%} إيجابية)")
        
        X = df.drop('target', axis=1)
        y = df['target']
        
        # إزالة الأعمدة غير الرقمية
        if 'date' in X.columns:
            X = X.drop('date', axis=1)
        
        self.feature_columns = X.columns.tolist()
        
        # تطبيع البيانات (مهم جداً لـ Gradient Boosting)
        X_scaled = self.scaler.fit_transform(X)
        
        # تقسيم زمني صحيح (لا خلط عشوائي)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        if verbose:
            print(f"📊 بيانات التدريب: {len(X_train)} عينة")
            print(f"📊 بيانات الاختبار: {len(X_test)} عينة")
            print("🧠 جاري تدريب نموذج Gradient Boosting المتقدم...")
        
        # تدريب مع معالجة عدم التوازن
        sample_weights = np.where(y_train == 1, 1/positive_ratio, 1/(1-positive_ratio))
        self.model.fit(X_train, y_train, sample_weight=sample_weights)
        
        # تقييم متقدم
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)
        
        # حساب نسبة الفوز عند عتبات ثقة مختلفة
        high_confidence_mask = y_proba > self.confidence_threshold
        if high_confidence_mask.sum() > 0:
            high_conf_precision = precision_score(
                y_test[high_confidence_mask], 
                y_pred[high_confidence_mask], 
                zero_division=0
            )
        else:
            high_conf_precision = 0
        
        if verbose:
            print(f"\n{'='*60}")
            print("✅ اكتمل التدريب بنجاح!")
            print(f"{'='*60}")
            print(f"📈 الدقة العامة (Accuracy): {accuracy:.2%}")
            print(f"🎯 دقة التنبؤات الإيجابية (Precision): {precision:.2%}")
            print(f"🏆 دقة الصفقات عالية الثقة (>{self.confidence_threshold:.0%}): {high_conf_precision:.2%}")
            print(f"📊 عدد الفرص عالية الثقة: {high_confidence_mask.sum()} من {len(y_test)}")
            
            if high_conf_precision > 0.65:
                print("\n🎉 ممتاز! النموذج جاهز للتداول بحذر")
            elif high_conf_precision > 0.55:
                print("\n✅ جيد - يحتاج لمراقبة دقيقة")
            else:
                print("\n⚠️ تحذير: يحتاج إلى مزيد من التحسين أو بيانات أكثر")
        
        self.is_trained = True
        return True
    
    def predict_signal(self, current_market_data, verbose=False):
        """
        اتخاذ قرار تداول مع تحليل شامل
        """
        if not self.is_trained:
            return "WAIT", 0.0, "النموذج غير مدرب"
        
        df = self.prepare_features(current_market_data)
        
        if len(df) == 0:
            return "WAIT", 0.0, "بيانات غير كافية"
        
        last_row = df.iloc[[-1]]
        
        try:
            last_row = last_row[self.feature_columns]
        except KeyError as e:
            return "WAIT", 0.0, f"أعمدة مفقودة: {e}"
        
        # تطبيع البيانات
        last_row_scaled = self.scaler.transform(last_row)
        
        # التنبؤ
        prediction = self.model.predict(last_row_scaled)[0]
        probability = self.model.predict_proba(last_row_scaled)[0][1]
        
        # تحليل أسباب القرار
        reasons = []
        
        # استخراج أهم العوامل
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # تحديد إشارة التداول
        if prediction == 1 and probability >= self.confidence_threshold:
            decision = "STRONG BUY"
            reasons.append("إشارة شراء قوية بثقة عالية")
        elif prediction == 1 and probability >= 0.60:
            decision = "BUY"
            reasons.append("إشارة شراء متوسطة الثقة")
        else:
            decision = "WAIT"
            reasons.append("الانتظار لفرصة أفضل")
        
        # إضافة ملاحظات من المؤشرات
        if 'trend_score' in df.columns:
            trend = df['trend_score'].iloc[-1]
            if trend > 0.8:
                reasons.append("اتجاه صاعد قوي")
            elif trend < 0.2:
                reasons.append("اتجاه هابط - احذر")
        
        if 'rsi_14' in df.columns:
            rsi = df['rsi_14'].iloc[-1]
            if rsi < 30:
                reasons.append("RSI تشبع بيعي (فرصة محتملة)")
            elif rsi > 70:
                reasons.append("RSI تشبع شرائي (احذر)")
        
        if verbose:
            print(f"\n{'='*60}")
            print("📊 تحليل الإشارة:")
            print(f"{'='*60}")
            for reason in reasons:
                print(f"  • {reason}")
            print(f"\n🎯 درجة الثقة: {probability:.2%}")
        
        return decision, probability, "; ".join(reasons)


# === اختبار النظام المتقدم ===
if __name__ == "__main__":
    print("="*60)
    print("🚀 نظام التداول الاحترافي بالذكاء الاصطناعي")
    print("="*60)
    
    # توليد بيانات واقعية
    np.random.seed(42)
    n_periods = 3000
    
    dates = pd.date_range(start="2022-01-01", periods=n_periods, freq='h')
    
    # محاكاة سوق حقيقي مع اتجاهات ودورات
    base_price = 100
    trend = np.sin(np.linspace(0, 4*np.pi, n_periods)) * 0.001  # دورة سوقية
    noise = np.random.randn(n_periods) * 0.015  # ضوضاء يومية
    returns = trend + noise + 0.0002  # عائد طفيف إيجابي
    
    close_prices = base_price * np.cumprod(1 + returns)
    high_prices = close_prices * (1 + np.abs(np.random.randn(n_periods)) * 0.012)
    low_prices = close_prices * (1 - np.abs(np.random.randn(n_periods)) * 0.012)
    open_prices = np.roll(close_prices, 1)  # استخدام roll بدلاً من shift
    open_prices[0] = close_prices[0]  # تصحيح أول قيمة
    volume = np.random.randint(1000, 5000, n_periods) * (close_prices / base_price)
    
    data = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })
    
    # تشغيل النظام
    trader = AdvancedMLTrader(confidence_threshold=0.75)
    
    success = trader.train(data)
    
    if success:
        print("\n" + "="*60)
        print("🔍 اختبار الإشارات اللحظية")
        print("="*60)
        
        # اختبار على آخر 200 شمعة
        recent_data = data.tail(200)
        signal, prob, analysis = trader.predict_signal(recent_data, verbose=True)
        
        print(f"\n📡 القرار النهائي: {signal}")
        print(f"📝 التحليل: {analysis}")
        
        if signal in ["BUY", "STRONG BUY"]:
            print("\n💰 توصيات إدارة المخاطر:")
            print("   • وقف الخسارة: 1.5% تحت سعر الدخول")
            print("   • هدف الربح الأول: 2.5%")
            print("   • هدف الربح الثاني: 4.0%")
            print("   • حجم الصفقة: 2-3% من رأس المال كحد أقصى")
        else:
            print("\n💤 النظام يفضل الانتظار - هذه أيضاً استراتيجية رابحة!")
