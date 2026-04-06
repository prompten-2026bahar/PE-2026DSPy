# DSPy ile E-postalardan Bilgi Çıkarma

Bu eğitim, DSPy kullanarak akıllı bir e-posta işleme sistemi oluşturmayı gösterir. Çeşitli e-posta türlerinden temel bilgileri otomatik olarak çıkarabilen, amaçlarını sınıflandırabilen ve verileri daha ileri işleme için yapılandırabilen bir sistem oluşturacağız.

## Ne İnşa Edeceksiniz

Bu eğitimin sonunda, aşağıdakileri yapabilen DSPy destekli bir e-posta işleme sistemine sahip olacaksınız:

- **E-posta türlerini sınıflandırma** (sipariş onayı, destek talebi, toplantı daveti vb.)
- **Temel varlıkları çıkarma** (tarihler, tutarlar, ürün adları, iletişim bilgileri)
- **Aciliyet düzeylerini** ve gerekli eylemleri belirleme
- **Çıkarılan verileri** tutarlı biçimlere yapılandırma
- **Birden fazla e-posta biçimini** dayanıklı şekilde ele alma

## Ön Koşullar

- DSPy modülleri ve imzaları hakkında temel anlayış
- Python 3.9+ kurulu olmalı
- OpenAI API anahtarı (veya desteklenen başka bir LLM’e erişim)

## Kurulum ve Hazırlık

```bash
pip install dspy
```

<details>
<summary>Önerilir: Perde arkasında neler olduğunu anlamak için MLflow Tracing kurun.</summary>

### MLflow DSPy Entegrasyonu

<a href="https://mlflow.org/">MLflow</a>, DSPy ile doğal olarak entegre olan ve açıklanabilirlik ile deney takibi sunan bir LLMOps aracıdır. Bu eğitimde, MLflow’u kullanarak istemleri ve optimizasyon ilerlemesini izler olarak görselleştirebilir, böylece DSPy’nin davranışını daha iyi anlayabilirsiniz. Aşağıdaki dört adımı izleyerek MLflow’u kolayca kurabilirsiniz.

![MLflow Trace](./mlflow-tracing-email-extraction.png)

1. MLflow’u kurun

```bash
%pip install mlflow>=3.0.0
```

2. Ayrı bir terminalde MLflow arayüzünü başlatın
```bash
mlflow ui --port 5000 --backend-store-uri sqlite:///mlruns.db
```

3. Notebook’u MLflow’a bağlayın
```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("DSPy")
```

4. İzlemeyi etkinleştirin.
```python
mlflow.dspy.autolog()
```


Entegrasyon hakkında daha fazla bilgi edinmek için [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) sayfasını da ziyaret edin.
</details>

## Adım 1: Veri Yapılarımızı Tanımlama

Öncelikle, e-postalardan çıkarmak istediğimiz bilgi türlerini tanımlayalım:

```python
import dspy
from typing import List, Optional, Literal
from datetime import datetime
from pydantic import BaseModel
from enum import Enum

class EmailType(str, Enum):
    ORDER_CONFIRMATION = "order_confirmation"
    SUPPORT_REQUEST = "support_request"
    MEETING_INVITATION = "meeting_invitation"
    NEWSLETTER = "newsletter"
    PROMOTIONAL = "promotional"
    INVOICE = "invoice"
    SHIPPING_NOTIFICATION = "shipping_notification"
    OTHER = "other"

class UrgencyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ExtractedEntity(BaseModel):
    entity_type: str
    value: str
    confidence: float
```

## Adım 2: DSPy İmzalarını Oluşturma

Şimdi e-posta işleme hattımız için imzaları tanımlayalım:

```python
class ClassifyEmail(dspy.Signature):
    """İçeriğine göre bir e-postanın türünü ve aciliyetini sınıflandır."""

    email_subject: str = dspy.InputField(desc="E-postanın konu satırı")
    email_body: str = dspy.InputField(desc="E-postanın ana içeriği")
    sender: str = dspy.InputField(desc="E-posta gönderici bilgisi")

    email_type: EmailType = dspy.OutputField(desc="Sınıflandırılan e-posta türü")
    urgency: UrgencyLevel = dspy.OutputField(desc="E-postanın aciliyet düzeyi")
    reasoning: str = dspy.OutputField(desc="Sınıflandırmaya dair kısa açıklama")

class ExtractEntities(dspy.Signature):
    """E-posta içeriğinden temel varlıkları ve bilgileri çıkar."""

    email_content: str = dspy.InputField(desc="Konu ve gövde dahil tam e-posta içeriği")
    email_type: EmailType = dspy.InputField(desc="Sınıflandırılmış e-posta türü")

    key_entities: list[ExtractedEntity] = dspy.OutputField(desc="Tür, değer ve güven skoru içeren çıkarılmış varlık listesi")
    financial_amount: Optional[float] = dspy.OutputField(desc="Bulunan parasal tutarlar (ör. '$99.99')")
    important_dates: list[str] = dspy.OutputField(desc="E-postada bulunan önemli tarihlerin listesi")
    contact_info: list[str] = dspy.OutputField(desc="Çıkarılan ilgili iletişim bilgileri")

class GenerateActionItems(dspy.Signature):
    """E-posta içeriği ve çıkarılan bilgiye göre hangi eylemlerin gerektiğini belirle."""

    email_type: EmailType = dspy.InputField()
    urgency: UrgencyLevel = dspy.InputField()
    email_summary: str = dspy.InputField(desc="E-posta içeriğinin kısa özeti")
    extracted_entities: list[ExtractedEntity] = dspy.InputField(desc="E-postada bulunan temel varlıklar")

    action_required: bool = dspy.OutputField(desc="Herhangi bir eylem gerekip gerekmediği")
    action_items: list[str] = dspy.OutputField(desc="Gerekli belirli eylemlerin listesi")
    deadline: Optional[str] = dspy.OutputField(desc="Varsa eylem için son tarih")
    priority_score: int = dspy.OutputField(desc="1-10 arası öncelik puanı")

class SummarizeEmail(dspy.Signature):
    """E-posta içeriğinin öz ve kısa bir özetini oluştur."""

    email_subject: str = dspy.InputField()
    email_body: str = dspy.InputField()
    key_entities: list[ExtractedEntity] = dspy.InputField()

    summary: str = dspy.OutputField(desc="E-postanın ana noktalarını anlatan 2-3 cümlelik özet")
```

## Adım 3: E-posta İşleme Modülünü Oluşturma

Şimdi ana e-posta işleme modülümüzü oluşturalım:

```python
class EmailProcessor(dspy.Module):
    """DSPy kullanan kapsamlı bir e-posta işleme sistemi."""

    def __init__(self):
        super().__init__()

        # İşleme bileşenlerimizi başlat
        self.classifier = dspy.ChainOfThought(ClassifyEmail)
        self.entity_extractor = dspy.ChainOfThought(ExtractEntities)
        self.action_generator = dspy.ChainOfThought(GenerateActionItems)
        self.summarizer = dspy.ChainOfThought(SummarizeEmail)

    def forward(self, email_subject: str, email_body: str, sender: str = ""):
        """Bir e-postayı işle ve yapılandırılmış bilgi çıkar."""

        # Adım 1: E-postayı sınıflandır
        classification = self.classifier(
            email_subject=email_subject,
            email_body=email_body,
            sender=sender
        )

        # Adım 2: Varlıkları çıkar
        full_content = f"Subject: {email_subject}\n\nFrom: {sender}\n\n{email_body}"
        entities = self.entity_extractor(
            email_content=full_content,
            email_type=classification.email_type
        )

        # Adım 3: Özet üret
        summary = self.summarizer(
            email_subject=email_subject,
            email_body=email_body,
            key_entities=entities.key_entities
        )

        # Adım 4: Eylemleri belirle
        actions = self.action_generator(
            email_type=classification.email_type,
            urgency=classification.urgency,
            email_summary=summary.summary,
            extracted_entities=entities.key_entities
        )

        # Adım 5: Sonuçları yapılandır
        return dspy.Prediction(
            email_type=classification.email_type,
            urgency=classification.urgency,
            summary=summary.summary,
            key_entities=entities.key_entities,
            financial_amount=entities.financial_amount,
            important_dates=entities.important_dates,
            action_required=actions.action_required,
            action_items=actions.action_items,
            deadline=actions.deadline,
            priority_score=actions.priority_score,
            reasoning=classification.reasoning,
            contact_info=entities.contact_info
        )
```

## Adım 4: E-posta İşleme Sistemini Çalıştırma

Şimdi e-posta işleme sistemimizi test etmek için basit bir fonksiyon oluşturalım:

```python
import os
def run_email_processing_demo():
    """E-posta işleme sisteminin gösterimi."""

    # DSPy'yi yapılandır
    lm = dspy.LM(model='openai/gpt-4o-mini')
    dspy.configure(lm=lm)
    os.environ["OPENAI_API_KEY"] = "<YOUR OPENAI KEY>"

    # E-posta işleyicimizi oluştur
    processor = EmailProcessor()

    # Test için örnek e-postalar
    sample_emails = [
        {
            "subject": "Order Confirmation #12345 - Your MacBook Pro is on the way!",
            "body": """Dear John Smith,

Thank you for your order! We're excited to confirm that your order #12345 has been processed.

Order Details:
- MacBook Pro 14-inch (Space Gray)
- Order Total: $2,399.00
- Estimated Delivery: December 15, 2024
- Tracking Number: 1Z999AA1234567890

If you have any questions, please contact our support team at support@techstore.com.

Best regards,
TechStore Team""",
            "sender": "orders@techstore.com"
        },
        {
            "subject": "URGENT: Server Outage - Immediate Action Required",
            "body": """Hi DevOps Team,

We're experiencing a critical server outage affecting our production environment.

Impact: All users unable to access the platform
Started: 2:30 PM EST

Please join the emergency call immediately: +1-555-123-4567

This is our highest priority.

Thanks,
Site Reliability Team""",
            "sender": "alerts@company.com"
        },
        {
            "subject": "Meeting Invitation: Q4 Planning Session",
            "body": """Hello team,

You're invited to our Q4 planning session.

When: Friday, December 20, 2024 at 2:00 PM - 4:00 PM EST
Where: Conference Room A

Please confirm your attendance by December 18th.

Best,
Sarah Johnson""",
            "sender": "sarah.johnson@company.com"
        }
    ]

    # Her e-postayı işle ve sonuçları göster
    print("🚀 E-posta İşleme Demosu")
    print("=" * 50)

    for i, email in enumerate(sample_emails):
        print(f"\n📧 E-POSTA {i+1}: {email['subject'][:50]}...")

        # E-postayı işle
        result = processor(
            email_subject=email["subject"],
            email_body=email["body"],
            sender=email["sender"]
        )

        # Temel sonuçları göster
        print(f"   📊 Tür: {result.email_type}")
        print(f"   🚨 Aciliyet: {result.urgency}")
        print(f"   📝 Özet: {result.summary}")

        if result.financial_amount:
            print(f"   💰 Tutar: ${result.financial_amount:,.2f}")

        if result.action_required:
            print(f"   ✅ Eylem Gerekli: Evet")
            if result.deadline:
                print(f"   ⏰ Son Tarih: {result.deadline}")
        else:
            print(f"   ✅ Eylem Gerekli: Hayır")

# Demoyu çalıştır
if __name__ == "__main__":
    run_email_processing_demo()
```

## Beklenen Çıktı
```
🚀 E-posta İşleme Demosu
==================================================

📧 E-POSTA 1: Order Confirmation #12345 - Your MacBook Pro is on...
   📊 Tür: order_confirmation
   🚨 Aciliyet: low
   📝 Özet: E-posta, John Smith'in #12345 numaralı MacBook Pro 14-inch Space Gray siparişini $2,399.00 toplam tutarla ve 15 Aralık 2024 tahmini teslim tarihiyle doğruluyor. Takip numarası ve müşteri desteği iletişim bilgileri de yer alıyor.
   💰 Tutar: $2,399.00
   ✅ Eylem Gerekli: Hayır

📧 E-POSTA 2: URGENT: Server Outage - Immediate Action Required...
   📊 Tür: other
   🚨 Aciliyet: critical
   📝 Özet: Site Reliability Team, saat 2:30 PM EST'de başlayan ve tüm kullanıcıların platforma erişimini engelleyen kritik bir sunucu kesintisi bildirdi. Sorunu çözmek için DevOps Team'in derhal acil çağrıya katılması istendi.
   ✅ Eylem Gerekli: Evet
   ⏰ Son Tarih: Hemen

📧 E-POSTA 3: Meeting Invitation: Q4 Planning Session...
   📊 Tür: meeting_invitation
   🚨 Aciliyet: medium
   📝 Özet: Sarah Johnson, ekibi 20 Aralık 2024 tarihinde 2:00 PM - 4:00 PM EST saatleri arasında Conference Room A'da yapılacak Q4 planlama oturumuna davet etti. Katılımcılardan 18 Aralık'a kadar katılımlarını doğrulamaları isteniyor.
   ✅ Eylem Gerekli: Evet
   ⏰ Son Tarih: December 18th
```

## Sonraki Adımlar

- **Daha fazla e-posta türü ekleyin** ve sınıflandırmayı geliştirin (bülten, promosyon vb.)
- **E-posta sağlayıcılarıyla entegrasyon ekleyin** (Gmail API, Outlook, IMAP)
- **Farklı LLM’lerle deney yapın** ve optimizasyon stratejileri deneyin
- **Uluslararası e-posta işleme için çok dilli destek ekleyin**
- **Programınızın performansını artırmak için optimizasyon yapın**
