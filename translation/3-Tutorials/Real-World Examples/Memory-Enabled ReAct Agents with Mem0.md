# DSPy ReAct ve Mem0 ile Bellek Etkin Ajanlar Geliştirme

Bu eğitim, DSPy’nin ReAct çerçevesi ile [Mem0](https://docs.mem0.ai/) bellek yeteneklerini birleştirerek etkileşimler arasında bilgi hatırlayabilen akıllı konuşma ajanlarının nasıl oluşturulacağını gösterir. Bağlamsal bilgiyi depolayabilen, geri getirebilen ve kullanarak kişiselleştirilmiş ve tutarlı yanıtlar verebilen ajanlar oluşturmayı öğreneceksiniz.

## Ne İnşa Edeceksiniz

Bu eğitimin sonunda, aşağıdakileri yapabilen bellek etkin bir ajana sahip olacaksınız:

- **Kullanıcı tercihlerini** ve geçmiş konuşmaları hatırlama
- Kullanıcılar ve konular hakkında **olgusal bilgileri depolama ve geri getirme**
- Kararları yönlendirmek ve kişiselleştirilmiş yanıtlar vermek için **belleği kullanma**
- Bağlam farkındalığıyla **karmaşık çok turlu konuşmaları** yönetme
- **Farklı bellek türlerini** yönetme (olgular, tercihler, deneyimler)

## Ön Koşullar

- DSPy ve ReAct ajanları hakkında temel anlayış
- Python 3.9+ kurulu olmalı
- Tercih ettiğiniz LLM sağlayıcısı için API anahtarları

## Kurulum ve Hazırlık

```bash
pip install dspy mem0ai
```

## Adım 1: Mem0 Entegrasyonunu Anlama

Mem0, yapay zekâ ajanları için anıları depolayabilen, arayabilen ve geri getirebilen bir bellek katmanı sunar. DSPy ile nasıl entegre edileceğini anlamakla başlayalım:

```python
import dspy
from mem0 import Memory
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

# Ortamı yapılandır
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Mem0 bellek sistemini başlat
config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            "temperature": 0.1
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small"
        }
    }
}
```

## Adım 2: Bellek Farkındalıklı Araçlar Oluşturma

Bellek sistemiyle etkileşime geçebilen araçlar oluşturalım:

```python
import datetime

class MemoryTools:
    """Mem0 bellek sistemiyle etkileşim kurmak için araçlar."""

    def __init__(self, memory: Memory):
        self.memory = memory

    def store_memory(self, content: str, user_id: str = "default_user") -> str:
        """Bilgiyi belleğe kaydet."""
        try:
            self.memory.add(content, user_id=user_id)
            return f"Kaydedilen anı: {content}"
        except Exception as e:
            return f"Anı kaydedilirken hata oluştu: {str(e)}"

    def search_memories(self, query: str, user_id: str = "default_user", limit: int = 5) -> str:
        """İlgili anıları ara."""
        try:
            results = self.memory.search(query, user_id=user_id, limit=limit)
            if not results:
                return "İlgili anı bulunamadı."

            memory_text = "Bulunan ilgili anılar:\n"
            for i, result in enumerate(results["results"]):
                memory_text += f"{i}. {result['memory']}\n"
            return memory_text
        except Exception as e:
            return f"Anılar aranırken hata oluştu: {str(e)}"

    def get_all_memories(self, user_id: str = "default_user") -> str:
        """Bir kullanıcı için tüm anıları getir."""
        try:
            results = self.memory.get_all(user_id=user_id)
            if not results:
                return "Bu kullanıcı için anı bulunamadı."

            memory_text = "Kullanıcı için tüm anılar:\n"
            for i, result in enumerate(results["results"]):
                memory_text += f"{i}. {result['memory']}\n"
            return memory_text
        except Exception as e:
            return f"Anılar alınırken hata oluştu: {str(e)}"

    def update_memory(self, memory_id: str, new_content: str) -> str:
        """Mevcut bir anıyı güncelle."""
        try:
            self.memory.update(memory_id, new_content)
            return f"Anı yeni içerikle güncellendi: {new_content}"
        except Exception as e:
            return f"Anı güncellenirken hata oluştu: {str(e)}"

    def delete_memory(self, memory_id: str) -> str:
        """Belirli bir anıyı sil."""
        try:
            self.memory.delete(memory_id)
            return "Anı başarıyla silindi."
        except Exception as e:
            return f"Anı silinirken hata oluştu: {str(e)}"

def get_current_time() -> str:
    """Geçerli tarih ve saati getir."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

## Adım 3: Bellekle Geliştirilmiş ReAct Ajanını Oluşturma

Şimdi belleği kullanabilen ana ReAct ajanımızı oluşturalım:

```python
class MemoryQA(dspy.Signature):
    """
    Sen yardımcı bir asistansın ve bellek yöntemlerine erişimin var.
    Kullanıcının girdisine her yanıt verdiğinde, bilgiyi belleğe kaydetmeyi unutma;
    böylece bunu daha sonra kullanabilirsin.
    """
    user_input: str = dspy.InputField()
    response: str = dspy.OutputField()

class MemoryReActAgent(dspy.Module):
    """Mem0 bellek yetenekleriyle geliştirilmiş bir ReAct ajanı."""

    def __init__(self, memory: Memory):
        super().__init__()
        self.memory_tools = MemoryTools(memory)

        # ReAct için araç listesini oluştur
        self.tools = [
            self.memory_tools.store_memory,
            self.memory_tools.search_memories,
            self.memory_tools.get_all_memories,
            get_current_time,
            self.set_reminder,
            self.get_preferences,
            self.update_preferences,
        ]

        # ReAct'i araçlarımızla başlat
        self.react = dspy.ReAct(
            signature=MemoryQA,
            tools=self.tools,
            max_iters=6
        )

    def forward(self, user_input: str):
        """Kullanıcı girdisini bellek farkındalıklı akıl yürütmeyle işle."""

        return self.react(user_input=user_input)

    def set_reminder(self, reminder_text: str, date_time: str = None, user_id: str = "default_user") -> str:
        """Kullanıcı için bir hatırlatıcı ayarla."""
        reminder = f"{date_time} için hatırlatıcı ayarlandı: {reminder_text}"
        return self.memory_tools.store_memory(
            f"HATIRLATICI: {reminder}", 
            user_id=user_id
        )

    def get_preferences(self, category: str = "general", user_id: str = "default_user") -> str:
        """Belirli bir kategori için kullanıcı tercihlerini getir."""
        query = f"kullanıcı tercihleri {category}"
        return self.memory_tools.search_memories(
            query=query,
            user_id=user_id
        )

    def update_preferences(self, category: str, preference: str, user_id: str = "default_user") -> str:
        """Kullanıcı tercihlerini güncelle."""
        preference_text = f"{category} için kullanıcı tercihi: {preference}"
        return self.memory_tools.store_memory(
            preference_text,
            user_id=user_id
        )
```

## Adım 4: Bellekle Geliştirilmiş Ajanı Çalıştırma

Bellek etkin ajanımızla etkileşim kurmak için basit bir arayüz oluşturalım:

```python
import time
def run_memory_agent_demo():
    """Bellekle geliştirilmiş ReAct ajanının gösterimi."""

    # DSPy'yi yapılandır
    lm = dspy.LM(model='openai/gpt-4o-mini')
    dspy.configure(lm=lm)

    # Bellek sistemini başlat
    memory = Memory.from_config(config)

    # Ajanımızı oluştur
    agent = MemoryReActAgent(memory)

    # Bellek yeteneklerini gösteren örnek konuşma
    print("🧠 Bellekle Geliştirilmiş ReAct Ajanı Demosu")
    print("=" * 50)

    conversations = [
        "Merhaba, ben Alice ve İtalyan yemeklerini seviyorum, özellikle de pasta carbonara.",
        "Ben Alice. Sabahları yaklaşık 7'de egzersiz yapmayı tercih ederim.",
        "Ben Alice. Yemek tercihlerim hakkında ne hatırlıyorsun?",
        "Ben Alice. Yarın markete gitmem için bana bir hatırlatıcı ayarla.",
        "Ben Alice. Egzersiz tercihlerim neler?",
        "Ben Alice. Ayrıca hafta sonları yürüyüş yapmaktan hoşlanırım.",
        "Ben Alice. Şimdiye kadar benim hakkımda ne biliyorsun?"
    ]

    for i, user_input in enumerate(conversations, 1):
        print(f"\n📝 Kullanıcı: {user_input}")

        try:
            response = agent(user_input=user_input)
            print(f"🤖 Ajan: {response.response}")
            time.sleep(1)

        except Exception as e:
            print(f"❌ Hata: {e}")

# Gösterimi çalıştır
if __name__ == "__main__":
    run_memory_agent_demo()
```

## Beklenen Çıktı
```
🧠 Bellekle Geliştirilmiş ReAct Ajanı Demosu
==================================================

📝 Kullanıcı: Merhaba, ben Alice ve İtalyan yemeklerini seviyorum, özellikle de pasta carbonara.

🤖 Ajan: Merhaba Alice! Tanıştığımıza memnun oldum! Pasta carbonara harika bir seçim. Onu yemeyi sevdiğin favori bir yer var mı, yoksa evde yapmaktan mı hoşlanıyorsun?

📝 Kullanıcı: Ben Alice. Sabahları yaklaşık 7'de egzersiz yapmayı tercih ederim.

🤖 Ajan: Tanıştığımıza memnun oldum, Alice! Sabahları yaklaşık 7'de egzersiz yapmayı tercih ettiğini not ettim. Sabah antrenmanlarınla ilgili soruların olursa ya da tavsiye istersen çekinmeden sorabilirsin!

📝 Kullanıcı: Ben Alice. Yemek tercihlerim hakkında ne hatırlıyorsun?

🤖 Ajan: İtalyan yemeklerini sevdiğini ve en sevdiğin yemeğin pasta carbonara olduğunu hatırlıyorum!

📝 Kullanıcı: Ben Alice. Yarın markete gitmem için bana bir hatırlatıcı ayarla.

🤖 Ajan: Alice, yarın markete gitmen için bir hatırlatıcı ayarladım.

📝 Kullanıcı: Ben Alice. Egzersiz tercihlerim neler?

🤖 Ajan: Egzersiz tercihin sabahları yaklaşık 7'de çalışmak.

📝 Kullanıcı: Ben Alice. Ayrıca hafta sonları yürüyüş yapmaktan hoşlanırım.

🤖 Ajan: Tanıştığımıza memnun oldum, Alice! Hafta sonları yürüyüş yapmaktan hoşlandığını duymak harika. Parkur önerilerine veya yürüyüş ipuçlarına ihtiyacın olursa çekinmeden sorabilirsin!

📝 Kullanıcı: Ben Alice. Şimdiye kadar benim hakkımda ne biliyorsun?

🤖 Ajan: Şimdiye kadar hafta sonları yürüyüş yapmaktan hoşlandığını, sabahları yaklaşık 7'de egzersiz yapmayı tercih ettiğini ve özellikle pasta carbonara olmak üzere İtalyan yemeklerini sevdiğini biliyorum. Ayrıca 7 Ekim 2023 için ayarlanmış market alışverişi hatırlatıcın var.
```

## Sonraki Adımlar

- **Veritabanlarıyla bellek kalıcılığı** uygulayın (PostgreSQL, MongoDB)
- Daha iyi düzen için **bellek kategorilendirme** ve etiketleme ekleyin
- Veri yönetimi için **bellek son kullanma politikaları** oluşturun
- Üretim uygulamaları için **çok kullanıcılı bellek izolasyonu** kurun
- **Bellek analitiği** ve içgörüler ekleyin
- Daha güçlü anlamsal arama için **vektör veritabanlarıyla entegrasyon** sağlayın
- Uzun vadeli depolama verimliliği için **bellek sıkıştırma** uygulayın

Bu eğitim, DSPy’nin ReAct çerçevesinin Mem0’un bellek yetenekleriyle nasıl geliştirilebileceğini ve böylece etkileşimler arasında bilgi öğrenip hatırlayabilen, bağlama duyarlı akıllı ajanlar oluşturulabildiğini gösterir. Bu da onları gerçek dünya uygulamaları için daha kullanışlı hâle getirir.
