#!/usr/bin/env python3
"""
نظام تداول حي متكامل
يجلب بيانات حقيقية من بينانس، يدرب النموذج، ويعطي إشارات لحظية
"""

import pandas as pd
import numpy as np
import requests
from advanced_trader import AdvancedMLTrader
import time
from datetime import datetime

class LiveTradingSystem:
    """نظام تداول حي يجلب البيانات ويعطي إشارات"""
    
    def __init__(self, symbol='BTCUSDT', interval='1h'):
        self.symbol = symbol
        self.interval = interval
        self.trader = None
        self.last_signal = None
        self.last_signal_time = None
        
    def fetch_live_data(self, limit=3000):
        """جلب بيانات حية من بينانس"""
        print(f"📡 جاري جلب بيانات {self.symbol}...")
        
        try:
            url = 'https://api.binance.com/api/v3/klines'
            params = {
                'symbol': self.symbol,
                'interval': self.interval,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if isinstance(data, dict) and 'code' in data:
                print(f"❌ خطأ من بينانس: {data['msg']}")
                return None
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            # تحويل البيانات
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            
            print(f"✅ تم جلب {len(df)} شمعة بنجاح")
            return df
            
        except Exception as e:
            print(f"❌ خطأ في جلب البيانات: {e}")
            return None
    
    def train_on_live_data(self):
        """تدريب النموذج على البيانات الحية"""
        data = self.fetch_live_data()
        
        if data is None or len(data) < 500:
            print("❌ البيانات غير كافية للتدريب")
            return False
        
        print(f"\n🧠 جاري تدريب النموذج على {len(data)} شمعة...")
        
        self.trader = AdvancedMLTrader(confidence_threshold=0.70)
        success = self.trader.train(data, verbose=True)
        
        if success:
            print("\n✅ التدريب اكتمل! النظام جاهز لإعطاء الإشارات")
        
        return success
    
    def get_current_signal(self):
        """الحصول على إشارة تداول حية"""
        if self.trader is None or not self.trader.is_trained:
            print("⚠️ النموذج غير مدرب بعد!")
            return None
        
        # جلب آخر 200 شمعة للتحليل
        data = self.fetch_live_data(limit=200)
        
        if data is None:
            return None
        
        signal, confidence, analysis = self.trader.predict_signal(
            data.tail(200), 
            verbose=True
        )
        
        # حفظ الإشارة
        self.last_signal = signal
        self.last_signal_time = datetime.now()
        
        return {
            'signal': signal,
            'confidence': confidence,
            'analysis': analysis,
            'time': self.last_signal_time,
            'price': data['close'].iloc[-1]
        }
    
    def run_continuous(self, check_interval=300):
        """تشغيل مستمر يفحص السوق كل فترة"""
        print("="*60)
        print("🚀 بدء نظام التداول الحي المستمر")
        print("="*60)
        print(f"الزوج: {self.symbol}")
        print(f"الفترة الزمنية: {self.interval}")
        print(f"فحص كل: {check_interval} ثانية")
        print("="*60)
        
        # تدريب أولي
        if not self.train_on_live_data():
            print("❌ فشل التدريب الأولي")
            return
        
        print("\n🔄 بدء المراقبة المستمرة... (اضغط Ctrl+C للإيقاف)")
        
        try:
            while True:
                print(f"\n{'='*60}")
                print(f"⏰ الفحص: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("="*60)
                
                result = self.get_current_signal()
                
                if result:
                    print(f"\n📡 الإشارة الحالية: {result['signal']}")
                    print(f"🎯 درجة الثقة: {result['confidence']:.2%}")
                    print(f"💰 السعر الحالي: ${result['price']:.2f}")
                    print(f"📝 التحليل: {result['analysis']}")
                    
                    if result['signal'] in ['BUY', 'STRONG BUY']:
                        print("\n" + "!"*60)
                        print("⚠️ تنبيه: إشارة شراء محتملة!")
                        print("!"*60)
                        print("توصيات إدارة المخاطر:")
                        print("  • وقف الخسارة: 1.5% تحت سعر الدخول")
                        print("  • هدف الربح: 2.5-4.0%")
                        print("  • حجم الصفقة: 2% كحد أقصى من رأس المال")
                        print("!"*60)
                    else:
                        print("\n💤 النظام يفضل الانتظار حالياً")
                
                print(f"\n⏳ الانتظار {check_interval} ثانية قبل الفحص التالي...")
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n\n🛑 أوقف المستخدم النظام")
            print(f"آخر إشارة: {self.last_signal} في {self.last_signal_time}")


# === التشغيل الرئيسي ===
if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     🚀 نظام التداول الاحترافي بالذكاء الاصطناعي          ║
    ║           نسخة التداول الحي المباشر                       ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # إعدادات النظام
    SYMBOL = 'BTCUSDT'  # الزوج الذي تريد التداول عليه
    INTERVAL = '1h'      # الفترة الزمنية (1h, 4h, 1d)
    CHECK_EVERY = 300    # فحص كل 5 دقائق
    
    # إنشاء النظام
    system = LiveTradingSystem(symbol=SYMBOL, interval=INTERVAL)
    
    # خيار 1: تشغيل مستمر
    print("اختر وضع التشغيل:")
    print("1. تشغيل مستمر (يفحص السوق باستمرار)")
    print("2. فحص لمرة واحدة")
    
    choice = input("\nأدخل اختيارك (1 أو 2): ").strip()
    
    if choice == '1':
        system.run_continuous(check_interval=CHECK_EVERY)
    else:
        # فحص لمرة واحدة
        print("\n🔄 جاري التدريب والفحص لمرة واحدة...")
        if system.train_on_live_data():
            result = system.get_current_signal()
            if result:
                print(f"\n{'='*60}")
                print("📊 النتيجة النهائية:")
                print(f"{'='*60}")
                print(f"الإشارة: {result['signal']}")
                print(f"الثقة: {result['confidence']:.2%}")
                print(f"السعر: ${result['price']:.2f}")
                print(f"التحليل: {result['analysis']}")
