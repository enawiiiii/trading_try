import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import ccxt
import time
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. محرك البيانات (Data Engine)
# ==========================================
class DataEngine:
    def __init__(self, symbol='BTC/USDT', timeframe='1h'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange = ccxt.binance()
        print(f"📡 [DataEngine] تهيئة الاتصال بـ {symbol}...")

    def fetch_data(self, limit=2000):
        """جلب البيانات التاريخية والحسابات الفورية"""
        try:
            bars = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=limit)
            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return self._add_indicators(df)
        except Exception as e:
            print(f"❌ خطأ في جلب البيانات: {e}")
            return None

    def _add_indicators(self, df):
        """إضافة مؤشرات المؤسسات والذكاء الاصطناعي"""
        # VWAP (مؤشر المؤسسات)
        df['vwap'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
        
        # Moving Averages
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
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * 2)
        
        # ATR (لتحديد التقلب)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # Target: هل السعر سيرتفع في الشمعة القادمة؟ (1 = نعم، 0 = لا)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # تنظيف القيم_nan
        df.dropna(inplace=True)
        return df

# ==========================================
# 2. عقل الذكاء الاصطناعي (AI Core)
# ==========================================
class AICore:
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=150, 
            learning_rate=0.1, 
            max_depth=5, 
            random_state=42
        )
        self.is_trained = False
        self.features = ['vwap', 'ema_50', 'ema_200', 'rsi', 'macd', 'signal_line', 'bb_std', 'atr']

    def train(self, df):
        """تدريب العقل على البيانات"""
        print("🧠 [AI Core] جاري تدريب النموذج على البيانات التاريخية...")
        X = df[self.features]
        y = df['target']
        
        # تقسيم البيانات (80% تدريب، 20% اختبار)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        self.model.fit(X_train, y_train)
        
        # تقييم الأداء
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"✅ [AI Core] اكتمل التدريب! دقة النموذج على البيانات الاختبارية: {acc:.2%}")
        self.is_trained = True
        return acc

    def predict(self, current_data):
        """التنبؤ بالخطوة التالية"""
        if not self.is_trained:
            return 0.5, "WAIT"
        
        # استخراج الميزات للشمعة الحالية
        features = current_data[self.features].iloc[-1:].values
        prob = self.model.predict_proba(features)[0][1] # احتمال الصعود
        
        decision = "WAIT"
        if prob > 0.65:
            decision = "BUY"
        elif prob < 0.35:
            decision = "SELL"
            
        return prob, decision

# ==========================================
# 3. مدير المخاطر الذكي (Dynamic Risk Manager)
# ==========================================
class RiskManager:
    def __init__(self, balance=10000):
        self.balance = balance
        self.risk_per_trade = 0.02 # مخاطرة 2% من الرصيد

    def calculate_position_size(self, entry_price, stop_loss_price, atr):
        """حجم الصفقة بناءً على التقلب (ATR)"""
        # وقف الخسارة الديناميكي: 2 * ATR
        dynamic_sl_distance = 2 * atr
        if entry_price == 0: return 0
        
        risk_amount = self.balance * self.risk_per_trade
        position_size = risk_amount / dynamic_sl_distance
        
        return min(position_size, self.balance / entry_price) # عدم تجاوز الرصيد

    def get_levels(self, entry_price, atr, direction='BUY'):
        """تحديد مستويات الدخول والخروج ديناميكياً"""
        sl_distance = 2 * atr
        tp_distance = 3 * atr # نسبة عائد 1:1.5
        
        if direction == 'BUY':
            sl = entry_price - sl_distance
            tp = entry_price + tp_distance
        else:
            sl = entry_price + sl_distance
            tp = entry_price - tp_distance
            
        return sl, tp

# ==========================================
# 4. محرك التنفيذ والمحاكاة (Execution Engine)
# ==========================================
class ExecutionEngine:
    def __init__(self, initial_balance=10000):
        self.balance = initial_balance
        self.position = None # {'type': 'BUY', 'entry': price, 'size': size}
        self.trades_log = []
        self.risk_manager = RiskManager(balance=initial_balance)

    def execute_signal(self, signal, prob, df_row):
        """تنفيذ الإشارة أو إغلاق الصفقة"""
        current_price = df_row['close']
        atr = df_row['atr']
        timestamp = df_row.name
        
        # إذا كان لدينا صفقة مفتوحة
        if self.position:
            pnl = 0
            if self.position['type'] == 'BUY':
                pnl = (current_price - self.position['entry']) * self.position['size']
            
            # تحديث الرصيد المؤقت
            current_equity = self.balance + pnl
            
            # شروط الخروج (Stop Loss / Take Profit ديناميكي)
            sl, tp = self.risk_manager.get_levels(self.position['entry'], atr, 'BUY')
            
            should_close = False
            reason = ""
            
            if current_price <= sl:
                should_close = True
                reason = "Stop Loss"
            elif current_price >= tp:
                should_close = True
                reason = "Take Profit"
            elif signal == "SELL": # إشارة عكس
                should_close = True
                reason = "Reverse Signal"

            if should_close:
                self.balance += pnl
                self.trades_log.append({
                    'time': timestamp,
                    'type': 'CLOSE',
                    'price': current_price,
                    'pnl': pnl,
                    'reason': reason,
                    'balance': self.balance
                })
                self.position = None
                print(f"💰 [Exec] إغلاق صفقة ({reason}): الربح/الخسارة = ${pnl:.2f} | الرصيد: ${self.balance:.2f}")

        # فتح صفقة جديدة إذا لم يكن هناك مركز وكانت الإشارة قوية
        if not self.position and signal in ['BUY', 'SELL'] and prob > 0.65:
            # تبسيط: سنركز على الشراء فقط في هذا النموذج الأولي للتجربة
            if signal == 'BUY':
                size = self.risk_manager.calculate_position_size(current_price, 0, atr)
                sl, tp = self.risk_manager.get_levels(current_price, atr, 'BUY')
                
                self.position = {
                    'type': 'BUY',
                    'entry': current_price,
                    'size': size,
                    'sl': sl,
                    'tp': tp
                }
                self.trades_log.append({
                    'time': timestamp,
                    'type': 'OPEN',
                    'price': current_price,
                    'signal_prob': prob,
                    'sl': sl,
                    'tp': tp
                })
                print(f"🚀 [Exec] فتح صفقة شراء @ ${current_price:.2f} | وقف خسارة: ${sl:.2f} | هدف: ${tp:.2f}")

# ==========================================
# 5. النظام المتكامل (Main System Orchestrator)
# ==========================================
class UnifiedTradingSystem:
    def __init__(self, symbol='BTC/USDT'):
        print("🚀 بدء تشغيل نظام التداول الموحد...")
        self.data_engine = DataEngine(symbol=symbol)
        self.ai_core = AICore()
        self.exec_engine = ExecutionEngine(initial_balance=10000) # رصيد تجريبي 10k
        self.symbol = symbol

    def run_backtest_simulation(self):
        """تشغيل محاكاة كاملة على البيانات التاريخية"""
        print("\n--- 🔄 بدء محاكاة التداول على البيانات التاريخية ---\n")
        
        # 1. جلب البيانات
        df = self.data_engine.fetch_data(limit=3000)
        if df is None: return
        
        print(f"📊 تم تحميل {len(df)} شمعة لـ {self.symbol}")
        
        # 2. تدريب العقل
        acc = self.ai_core.train(df)
        
        # 3. محاكاة التداول خطوة بخطوة (Walk-Forward)
        # نبدأ من بعد فترة التدريب
        start_idx = int(len(df) * 0.8)
        simulation_df = df.iloc[start_idx:].copy()
        
        print(f"⏳ جاري محاكاة التداول على {len(simulation_df)} شمعة لاحقة...\n")
        
        for index, row in simulation_df.iterrows():
            # الحصول على تنبؤ
            prob, signal = self.ai_core.predict(row.to_frame().T)
            
            # تنفيذ الإشارة
            self.exec_engine.execute_signal(signal, prob, row)
            
            # (اختياري) طباعة حالة كل 100 شمعة لتقليل الضوضاء
            if len(self.exec_engine.trades_log) % 5 == 0 and len(self.exec_engine.trades_log) > 0:
                 pass # يمكن تفعيل logs مفصلة هنا

        self.generate_report()

    def generate_report(self):
        """تقرير أداء شامل"""
        print("\n" + "="*50)
        print("📊 تقرير أداء النظام الموحد")
        print("="*50)
        
        trades = self.exec_engine.trades_log
        closed_trades = [t for t in trades if t['type'] == 'CLOSE']
        
        if not closed_trades:
            print("⚠️ لم يتم إغلاق أي صفقات بعد.")
            return

        total_pnl = sum(t['pnl'] for t in closed_trades)
        wins = [t for t in closed_trades if t['pnl'] > 0]
        losses = [t for t in closed_trades if t['pnl'] <= 0]
        
        win_rate = (len(wins) / len(closed_trades)) * 100
        final_balance = self.exec_engine.balance
        
        print(f"💰 الرصيد الابتدائي: $10,000")
        print(f"💵 الرصيد النهائي: ${final_balance:.2f}")
        print(f"📈 صافي الربح/الخسارة: ${total_pnl:.2f} ({(total_pnl/10000)*100:.2f}%)")
        print(f"🏆 عدد الصفقات المغلقة: {len(closed_trades)}")
        print(f"✅ صفقات رابحة: {len(wins)}")
        print(f"❌ صفقات خاسرة: {len(losses)}")
        print(f"🎯 نسبة الفوز: {win_rate:.2f}%")
        
        if len(wins) > 0 and len(losses) > 0:
            avg_win = sum(t['pnl'] for t in wins) / len(wins)
            avg_loss = abs(sum(t['pnl'] for t in losses) / len(losses))
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
            print(f"⚖️ متوسط الربح: ${avg_win:.2f} | متوسط الخسارة: ${avg_loss:.2f}")
            print(f"📊 معامل الربح (Profit Factor): {profit_factor:.2f}")
        
        print("="*50)
        if win_rate > 50 and total_pnl > 0:
            print("🌟 النتيجة: ممتازة! النظام جاهز للمراقبة على حساب تجريبي حي.")
        elif total_pnl > 0:
            print("⚠️ النتيجة: مقبولة ولكن تحتاج تحسين فلترة الدخول.")
        else:
            print("🛑 النتيجة: خاسرة. يحتاج إعادة ضبط المعايير أو تغيير الاستراتيجية.")

# ==========================================
# التشغيل الرئيسي
# ==========================================
if __name__ == "__main__":
    # تشغيل النظام المتكامل
    system = UnifiedTradingSystem(symbol='BTC/USDT')
    system.run_backtest_simulation()
