"""
=============================================================================
🎰 تطبيق اليانصيب الأردني - الإصدار الاحترافي v8.0
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import requests
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import warnings
import os
import sys

warnings.filterwarnings('ignore')

# =============================================================================
# 🔧 إعدادات التطبيق
# =============================================================================

class Config:
    """إعدادات مركزية للتطبيق"""
    
    APP_VERSION = "8.0.0 PRO"
    APP_NAME = "Jordan Lottery AI Pro"
    
    # المسارات
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    
    # البيانات
    GITHUB_URL = "https://raw.githubusercontent.com/MohamedOmariJo/omari/main/250.xlsx"
    BACKUP_FILE = os.path.join(DATA_DIR, "history.xlsx")
    
    # نطاق الأرقام
    MIN_NUMBER = 1
    MAX_NUMBER = 32
    DEFAULT_TICKET_SIZE = 6
    
    # الذاكرة المؤقتة
    CACHE_TTL = 3600

# =============================================================================
# استيراد المكونات
# =============================================================================

# إضافة المسار للمجلدات
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# استيراد مع معالجة الأخطاء
def safe_import(module_name, class_name=None):
    """استيراد آمن للمكونات"""
    try:
        if class_name:
            module = __import__(module_name, fromlist=[class_name])
            return getattr(module, class_name)
        else:
            return __import__(module_name)
    except ImportError as e:
        st.warning(f"⚠️ لم يتم تحميل {module_name}.{class_name if class_name else ''}: {e}")
        return None

# استيراد المكونات
AdvancedAnalyzer = safe_import('core.analyzer', 'AdvancedAnalyzer')
AdvancedValidator = safe_import('core.validator', 'AdvancedValidator')
SmartGenerator = safe_import('core.generator', 'SmartGenerator')
LotteryPredictor = safe_import('core.models', 'LotteryPredictor')
RecommendationEngine = safe_import('core.models', 'RecommendationEngine')
DatabaseManager = safe_import('core.database', 'DatabaseManager')
NotificationSystem = safe_import('core.notifications', 'NotificationSystem')

# استيراد الأدوات
logger_module = safe_import('utils.logger')
if logger_module and hasattr(logger_module, 'logger'):
    logger = logger_module.logger
else:
    # logger بديل
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('lottery')

# =============================================================================
# 1. تحميل البيانات
# =============================================================================

@st.cache_data(ttl=Config.CACHE_TTL, show_spinner=True)
def load_data_with_retry():
    """تحميل البيانات مع إعادة المحاولة"""
    try:
        # محاولة التحميل من الإنترنت
        try:
            response = requests.get(Config.GITHUB_URL, timeout=15)
            response.raise_for_status()
            df = pd.read_excel(io.BytesIO(response.content))
            source = "GitHub"
            
        except requests.RequestException:
            # المحاولة من النسخة الاحتياطية
            try:
                df = pd.read_excel(Config.BACKUP_FILE)
                source = "Local Backup"
            except FileNotFoundError:
                return None, "❌ لم يتم العثور على ملف البيانات"
            except Exception as e:
                return None, f"❌ خطأ في قراءة الملف: {e}"
        
        # تنظيف البيانات
        df = validate_and_clean_data(df)
        
        if df.empty:
            return None, "❌ لا توجد سحوبات صالحة في البيانات"
        
        return df, f"✅ تم تحميل {len(df)} سحب من {source}"
        
    except Exception as e:
        return None, f"❌ خطأ غير متوقع: {e}"

def validate_and_clean_data(df):
    """تنظيف وتحقق من جودة البيانات"""
    # البحث عن أعمدة الأرقام
    number_cols = []
    for col in df.columns:
        if isinstance(col, str) and col.upper().startswith('N'):
            number_cols.append(col)
    
    if len(number_cols) < 6:
        # البحث عن أي أعمدة تحتوي على أرقام
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                number_cols.append(col)
    
    if len(number_cols) < 6:
        return pd.DataFrame()
    
    number_cols = number_cols[:6]
    
    # تحويل إلى أرقام
    df[number_cols] = df[number_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=number_cols, inplace=True)
    
    # دمج الأرقام
    df['numbers'] = df[number_cols].values.tolist()
    df['numbers'] = df['numbers'].apply(
        lambda x: sorted([int(n) for n in x if Config.MIN_NUMBER <= n <= Config.MAX_NUMBER])
    )
    
    # إزالة غير الصالحة
    df = df[df['numbers'].apply(len) == Config.DEFAULT_TICKET_SIZE].copy()
    
    # إضافة معلومات
    df['draw_id'] = range(1, len(df) + 1)
    df['date'] = [f"السحب {i}" for i in df['draw_id']]
    
    return df.reset_index(drop=True)

# =============================================================================
# 2. الواجهة الرئيسية
# =============================================================================

def main():
    """الواجهة الرئيسية للتطبيق"""
    
    # إعداد الصفحة
    st.set_page_config(
        page_title=Config.APP_NAME,
        page_icon="🎰",
        layout="wide"
    )
    
    # CSS مخصص
    st.markdown("""
    <style>
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 12px 24px;
            font-weight: bold;
        }
        .ball {
            display: inline-block;
            width: 50px;
            height: 50px;
            line-height: 50px;
            text-align: center;
            border-radius: 50%;
            color: white;
            font-weight: bold;
            margin: 5px;
            font-size: 18px;
        }
        .hot { background: linear-gradient(135deg, #ff6b6b, #ee5a52); }
        .cold { background: linear-gradient(135deg, #4ecdc4, #44a08d); }
        .neutral { background: linear-gradient(135deg, #ffeaa7, #fdcb6e); color: #2d3436; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title(f"🎰 {Config.APP_NAME}")
    st.markdown(f"**الإصدار:** {Config.APP_VERSION}")
    st.markdown("---")
    
    # تحميل البيانات
    if 'data_loaded' not in st.session_state:
        with st.spinner('🔄 جاري تحميل البيانات...'):
            df, msg = load_data_with_retry()
            
            if df is None:
                st.error(msg)
                st.stop()
            
            # حفظ البيانات
            st.session_state.df = df
            st.session_state.data_loaded = True
            
            # تهيئة المحلل إذا كان متاحاً
            if AdvancedAnalyzer:
                try:
                    analyzer = AdvancedAnalyzer(df)
                    st.session_state.analyzer = analyzer
                except Exception as e:
                    st.warning(f"⚠️ لم يتم تحميل المحلل المتقدم: {e}")
            
            st.success(msg)
    
    # الوصول للبيانات
    df = st.session_state.df
    
    # الشريط الجانبي
    with st.sidebar:
        st.header("📊 الإحصائيات")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("إجمالي السحوبات", len(df))
        with col2:
            st.metric("آخر سحب", f"#{df.iloc[-1]['draw_id']}")
        
        st.markdown("---")
        
        # عرض الأرقام الشائعة
        if hasattr(st.session_state, 'analyzer') and hasattr(st.session_state.analyzer, 'hot'):
            analyzer = st.session_state.analyzer
            st.subheader("🔥 الأرقام الساخنة")
            hot_nums = sorted(list(analyzer.hot))[:6]
            cols = st.columns(6)
            for i, num in enumerate(hot_nums):
                with cols[i]:
                    st.markdown(f'<div class="ball hot">{num}</div>', unsafe_allow_html=True)
    
    # التبويبات الرئيسية
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 الرئيسية", "🎰 توليد تذاكر", "🔍 فحص تذاكر", "📈 تحليلات"])
    
    with tab1:
        st.header("🏠 الصفحة الرئيسية")
        
        # آخر سحب
        last_draw = df.iloc[-1]
        st.subheader(f"🎱 آخر سحب (#{last_draw['draw_id']})")
        
        # عرض أرقام آخر سحب
        cols = st.columns(6)
        numbers = sorted(last_draw['numbers'])
        for i, num in enumerate(numbers):
            with cols[i]:
                # تحديد اللون
                if hasattr(st.session_state, 'analyzer'):
                    analyzer = st.session_state.analyzer
                    if num in analyzer.hot:
                        ball_class = "hot"
                    elif num in analyzer.cold:
                        ball_class = "cold"
                    else:
                        ball_class = "neutral"
                else:
                    ball_class = "neutral"
                
                st.markdown(f'<div class="ball {ball_class}">{num}</div>', unsafe_allow_html=True)
        
        # إحصائيات
        st.subheader("📈 إحصائيات سريعة")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_sum = sum(last_draw['numbers'])
            st.metric("مجموع الأرقام", total_sum)
        
        with col2:
            odd_count = sum(1 for n in numbers if n % 2)
            st.metric("فردي/زوجي", f"{odd_count}/{6-odd_count}")
        
        with col3:
            avg_sum = np.mean([sum(nums) for nums in df['numbers']])
            st.metric("المتوسط التاريخي", round(avg_sum, 1))
        
        # رسم بياني بسيط
        st.subheader("📊 تطور المجموع")
        
        if len(df) > 10:
            recent_df = df.tail(20).copy()
            recent_df['sum'] = recent_df['numbers'].apply(sum)
            
            fig = px.line(recent_df, x='draw_id', y='sum', 
                         title='تطور مجموع الأرقام في آخر 20 سحب',
                         markers=True)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("🎰 توليد التذاكر")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ الإعدادات")
            
            ticket_count = st.slider("عدد التذاكر", 1, 50, 10)
            strategy = st.selectbox("استراتيجية التوليد", 
                                   ["عشوائي", "متوازن", "تركيز على الساخن"])
            
            if st.button("🚀 توليد تذاكر", use_container_width=True):
                if SmartGenerator and hasattr(st.session_state, 'analyzer'):
                    try:
                        generator = SmartGenerator(st.session_state.analyzer)
                        tickets = generator.generate_tickets(ticket_count, 6, {})
                        st.session_state.generated_tickets = tickets
                        st.success(f"✅ تم توليد {len(tickets)} تذكرة")
                    except Exception as e:
                        st.error(f"❌ خطأ في التوليد: {e}")
                else:
                    # توليد بسيط
                    tickets = []
                    for _ in range(ticket_count):
                        ticket = np.random.choice(range(1, 33), 6, replace=False)
                        tickets.append(sorted(ticket.tolist()))
                    st.session_state.generated_tickets = tickets
                    st.success(f"✅ تم توليد {len(tickets)} تذكرة (نسخة بسيطة)")
        
        with col2:
            st.subheader("📋 النتائج")
            
            if 'generated_tickets' in st.session_state:
                tickets = st.session_state.generated_tickets
                
                for i, ticket in enumerate(tickets[:10]):  # عرض أول 10 تذاكر فقط
                    with st.expander(f"🎫 التذكرة #{i+1}", expanded=(i < 3)):
                        # عرض الأرقام
                        row_cols = st.columns(6)
                        for j, num in enumerate(ticket):
                            with row_cols[j]:
                                # تحديد اللون
                                if hasattr(st.session_state, 'analyzer'):
                                    analyzer = st.session_state.analyzer
                                    if num in analyzer.hot:
                                        ball_class = "hot"
                                    elif num in analyzer.cold:
                                        ball_class = "cold"
                                    else:
                                        ball_class = "neutral"
                                else:
                                    ball_class = "neutral"
                                
                                st.markdown(f'<div class="ball {ball_class}">{num}</div>', unsafe_allow_html=True)
                        
                        # معلومات سريعة
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            st.metric("المجموع", sum(ticket))
                        with col_info2:
                            odd = sum(1 for n in ticket if n % 2)
                            st.metric("فردي", odd)
    
    with tab3:
        st.header("🔍 فحص التذاكر")
        
        st.write("أدخل أرقام تذكرتك للتحقق من التطابقات التاريخية")
        
        # إدخال الأرقام
        input_method = st.radio("طريقة الإدخال", ["كتابة يدوية", "اختيار من قائمة"])
        
        if input_method == "كتابة يدوية":
            ticket_input = st.text_input("الأرقام (مفصولة بفواصل)", "5, 12, 18, 23, 27, 30")
            numbers = []
            for part in ticket_input.split(','):
                part = part.strip()
                if part.isdigit():
                    num = int(part)
                    if 1 <= num <= 32:
                        numbers.append(num)
            numbers = sorted(numbers[:6])
        else:
            # اختيار من قائمة
            cols = st.columns(6)
            numbers = []
            for i in range(6):
                with cols[i]:
                    num = st.selectbox(f"الرقم {i+1}", range(1, 33), key=f"num_{i}")
                    numbers.append(num)
            numbers = sorted(numbers)
        
        # عرض التذكرة المدخلة
        if numbers:
            st.subheader("🎫 تذكرتك")
            cols = st.columns(6)
            for i, num in enumerate(numbers):
                with cols[i]:
                    st.markdown(f'<div class="ball neutral">{num}</div>', unsafe_allow_html=True)
        
        if st.button("🔍 فحص التذكرة", use_container_width=True):
            if len(numbers) < 6:
                st.error("❌ الرجاء إدخال 6 أرقام")
            else:
                with st.spinner("جاري البحث في السحوبات السابقة..."):
                    ticket_set = set(numbers)
                    matches = []
                    
                    for _, row in df.iterrows():
                        draw_set = set(row['numbers'])
                        match_count = len(ticket_set & draw_set)
                        
                        if match_count >= 3:
                            matches.append({
                                'draw_id': row['draw_id'],
                                'date': row['date'],
                                'match_count': match_count,
                                'matching_numbers': sorted(list(ticket_set & draw_set))
                            })
                    
                    if matches:
                        st.success(f"🎉 وجدنا {len(matches)} تطابق!")
                        
                        # عرض أفضل 5 تطابقات
                        matches.sort(key=lambda x: x['match_count'], reverse=True)
                        
                        for match in matches[:5]:
                            with st.expander(f"السحب #{match['draw_id']} - {match['match_count']} مطابقة"):
                                st.write(f"**التاريخ:** {match['date']}")
                                st.write(f"**الأرقام المطابقة:** {match['matching_numbers']}")
                    else:
                        st.warning("😔 لا يوجد أي تطابقات (3+ أرقام) في السجل التاريخي")
    
    with tab4:
        st.header("📈 التحليلات المتقدمة")
        
        st.info("تحليلات وإحصائيات متقدمة")
        
        # توزيع الأرقام
        st.subheader("📊 توزيع تكرار الأرقام")
        
        # حساب التكرار
        all_numbers = []
        for nums in df['numbers']:
            all_numbers.extend(nums)
        
        freq = pd.Series(all_numbers).value_counts().sort_index()
        
        fig = px.bar(x=freq.index, y=freq.values,
                    labels={'x': 'الرقم', 'y': 'التكرار'},
                    title='عدد مرات ظهور كل رقم')
        st.plotly_chart(fig, use_container_width=True)
        
        # توزيع المجاميع
        st.subheader("📈 توزيع مجموع الأرقام")
        
        sums = [sum(nums) for nums in df['numbers']]
        
        fig2 = px.histogram(x=sums, nbins=30,
                           labels={'x': 'المجموع', 'y': 'التكرار'},
                           title='توزيع مجموع الأرقام في السحوبات')
        st.plotly_chart(fig2, use_container_width=True)
    
    # تذييل الصفحة
    st.markdown("---")
    st.caption(f"© 2026 - {Config.APP_NAME} v{Config.APP_VERSION}")

# =============================================================================
# تشغيل التطبيق
# =============================================================================

if __name__ == "__main__":
    main()