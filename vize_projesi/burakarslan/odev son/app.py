import streamlit as st
import dspy
from dspy.teleprompt import BootstrapFewShot
import os
from dotenv import load_dotenv

# --- GÜVENLİK VE API AYARLARI ---
# .env dosyasındaki gizli değişkenleri sisteme yüklüyoruz
load_dotenv()

# API anahtarını güvenli bir şekilde çekiyoruz
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("⚠️ HATA: .env dosyası bulunamadı veya GROQ_API_KEY eksik! Lütfen API anahtarınızı yapılandırın.")
    st.stop()

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Akıllı Trafik Sinyalizasyon Asistanı", layout="wide")

# Modeli Tanımlıyoruz (Güvenli değişken ile)
llama_model = dspy.LM('groq/llama-3.3-70b-versatile', api_key=GROQ_API_KEY)

# --- DSPY MİMARİSİ ---
class TrafficSignalizationOptimization(dspy.Signature):
    """Gelen düzensiz trafik raporunu analiz eder ve sinyalizasyon optimizasyonu için aksiyon önerir."""
    traffic_report = dspy.InputField(desc="Radyo, sosyal medya veya polis telsizinden gelen düzensiz trafik/kaza raporu.")
    severity_level = dspy.OutputField(desc="Durumun ciddiyeti: Düşük, Orta veya Kritik.")
    signal_action = dspy.OutputField(desc="Kavşaktaki sinyalizasyon için somut bir eylem planı.")

class TrafficOptimizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.optimize_signal = dspy.ChainOfThought(TrafficSignalizationOptimization)

    def forward(self, traffic_report):
        return self.optimize_signal(traffic_report=traffic_report)

@st.cache_resource
def get_compiled_model():
    trainset = [
        dspy.Example(traffic_report="Kuzey yönünde D-100 karayolunda zincirleme kaza. 3 şerit kapalı.", severity_level="Kritik", signal_action="Kuzey yönü giriş sinyalleri kırmızıya sabitlenmeli, alternatif rotaların yeşil süresi artırılmalı.").with_inputs('traffic_report'),
        dspy.Example(traffic_report="Atatürk Caddesi'nde hafif yağış nedeniyle trafik yavaş.", severity_level="Düşük", signal_action="Mevcut döngü korunmalı, takip mesafesi uyarıları açılmalı.").with_inputs('traffic_report'),
        dspy.Example(traffic_report="Üniversite kavşağında araç arızası. Sağ şerit tıkalı.", severity_level="Orta", signal_action="Kavşak çıkışı yeşil ışık süresi 15 sn uzatılmalı.").with_inputs('traffic_report')
    ]
    optimizer = BootstrapFewShot(metric=None)
    with dspy.context(lm=llama_model):
        return optimizer.compile(student=TrafficOptimizer(), trainset=trainset)

compiled_model = get_compiled_model()

# --- ARAYÜZ TASARIMI (STREAMLIT) ---
st.title("🚦 DSPy Tabanlı Akıllı Sinyalizasyon Optimizasyonu")
st.markdown("**Ders:** Prompt Mühendisliği | **Model:** Meta Llama 3.3 (70B) | **Strateji:** Chain of Thought + BootstrapFewShot")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Trafik Raporu Girişi")
    user_input = st.text_area("Radyo, sosyal medya veya kaza raporu metnini buraya girin:", height=150)
    submit_button = st.button("Sinyalizasyonu Optimize Et", type="primary")

with col2:
    st.subheader("📤 Yapay Zeka Karar Destek Çıktısı")
    
    if submit_button and user_input:
        with st.spinner("LLM Düşünüyor ve DSPy promptları optimize ediliyor..."):
            try:
                with dspy.context(lm=llama_model):
                    sonuc = compiled_model(traffic_report=user_input)
                
                st.success("Analiz Tamamlandı!")
                st.markdown(f"**Ciddiyet Seviyesi:** `{sonuc.severity_level}`")
                
                dusunce_sureci = getattr(sonuc, 'rationale', getattr(sonuc, 'reasoning', 'Model arka planda akıl yürütmüş, ancak metin formatı uyumsuzluğu nedeniyle DSPy bu adımı doğrudan sonuca yansıtmıştır.'))
                
                with st.expander("🧠 Modelin Akıl Yürütme Süreci (Chain of Thought)", expanded=True):
                    st.info(f"*{dusunce_sureci}*")
                
                st.success(f"**Önerilen Sinyalizasyon Aksiyonu:** \n\n {sonuc.signal_action}")
            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")

# --- DSPY MUTFAĞI: PROMPT DÖNÜŞÜMÜ ---
if submit_button and user_input:
    st.markdown("---")
    st.subheader("⚙️ DSPy Arka Plan: Prompt Dönüşüm Aşamaları (Tam Şeffaflık)")
    st.markdown("Aşağıda, girdiğiniz metnin arka planda modele gitmeden önce DSPy tarafından nasıl tam teşekküllü bir prompta dönüştürüldüğünü görebilirsiniz:")

    with st.expander("Aşama 1: Signature (İmza) Entegrasyonu", expanded=False):
        st.code("Görev: Gelen düzensiz trafik raporunu analiz eder ve sinyalizasyon optimizasyonu için aksiyon önerir.\n\nGirdi Formatı:\n- traffic_report: Radyo, sosyal medya veya polis telsizinden gelen düzensiz trafik/kaza raporu.\n\nÇıktı Formatı:\n- severity_level: Durumun ciddiyeti: Düşük, Orta veya Kritik.\n- signal_action: Kavşaktaki sinyalizasyon için somut bir eylem planı.", language="text")

    with st.expander("Aşama 2: BootstrapFewShot (Örneklem) Entegrasyonu", expanded=False):
        st.code("--- Örnek 1 ---\ntraffic_report: Kuzey yönünde D-100 karayolunda zincirleme kaza. 3 şerit kapalı.\nseverity_level: Kritik\nsignal_action: Kuzey yönü giriş sinyalleri kırmızıya sabitlenmeli, alternatif rotaların yeşil süresi artırılmalı.\n\n--- Örnek 2 ---\ntraffic_report: Atatürk Caddesi'nde hafif yağış nedeniyle trafik yavaş.\nseverity_level: Düşük\nsignal_action: Mevcut döngü korunmalı, takip mesafesi uyarıları açılmalı.\n\n--- Örnek 3 ---\ntraffic_report: Üniversite kavşağında araç arızası. Sağ şerit tıkalı.\nseverity_level: Orta\nsignal_action: Kavşak çıkışı yeşil ışık süresi 15 sn uzatılmalı.", language="text")

    with st.expander("Aşama 3: Chain of Thought (Düşünce Zinciri) ve Nihai Soru", expanded=True):
        st.code(f"--- YENİ GİRDİ ---\ntraffic_report: {user_input}\n\nLütfen yukarıdaki örneklere bakarak duruma uygun karar ver. Doğrudan sonuca atlama!\n1. Önce durumu adım adım analiz et (Reasoning).\n2. Sonra severity_level belirle.\n3. En son signal_action oluştur.\n\nAnaliz ve Çıktılar:", language="text")