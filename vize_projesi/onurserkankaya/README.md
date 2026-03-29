[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Y_zesxa5UmRrmgx1ZvcmnQvawF2EDxyx?usp=sharing)
# 🛡️ DSPy Guard: Yapay Zeka Destekli Otonom E-posta Siber Tehdit Analizi

DSPy Guard, şüpheli e-postaları (phishing/oltalama) siber güvenlik perspektifinden inceleyen, risk analizi yapan ve profesyonel raporlar sunan **DSPy Framework** tabanlı bir otonom analiz boru hattıdır.

---

## 🚀 Proje Ne Yapıyor?
Kullanıcı sisteme bir `.eml` dosyası sağladığında, sistem şu dört aşamalı süreci gerçekleştirir:
1. **Veri Ayrıştırma (Parsing):** E-posta meta verilerini deşifre ederek temiz bir veri yapısına dönüştürür.
2. **Siber Tehdit Analizi:** Sosyal mühendislik taktiklerini ve şüpheli bağlantıları tespit eder.
3. **Yanlış Pozitif Filtreleme:** Yasal kampanya maillerini oltalama saldırılarından ayırt eder.
4. **Profesyonel Raporlama:** Bulguları 1-10 arası bir risk skoruyla görsel bir tabloda sunur.

---

## 🛠️ Teknik Mimari
Sistem, geleneksel prompt mühendisliği yerine **DSPy** kütüphanesinin deklaratif programlama mantığını kullanır:

- **Hibrit LLM Desteği:** Yerel modeller (**Ollama/Llama 3.2**) ve **Gemini 1.5 Flash API** ile entegre çalışır.
- **Chain of Thought (CoT):** Model doğrudan tahmin yerine, siber güvenlik mantığı yürüterek sonuç üretir.
- **Regex Sanitization:** LLM çıktılarındaki sistem kalıntılarını düzenli ifadelerle temizler.

---

## 🤖 Kullanılan Modüller

| Modül | Görevi |
| :--- | :--- |
| **EML Parser** | `.eml` dosyalarını temiz metin haline getirir. |
| **DSPy Signature** | Güvenlik uzmanı rolünü ve kuralları belirler. |
| **Logic Engine** | Risk skoru ve analiz üretir. |
| **Reporter** | Bulguları ASCII tablolarına dönüştürür. |

---

## 📁 Proje Yapısı
- **Hücre 1:** Kütüphane kurulumları ve altyapı hazırlığı.
- **Hücre 2:** Gemini API bağlantısı ve DSPy motorunun inşası.
- **Hücre 3:** Operasyonel analiz, Regex filtreleme ve raporlama.

---
> *Bu proje Yapay Zeka Yüksek Lisansı "Prompt Mühendisliği" dersi kapsamında geliştirilmiştir.*
