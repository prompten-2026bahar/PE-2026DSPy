# DSPy ile Yapay Zeka Ajanları Oluşturma

Bu eğitimde, DSPy kullanarak nasıl yapay zeka ajanları (AI agents) oluşturabileceğinizi adım adım göstereceğiz. Yapay zeka ajanları; çevresini otonom olarak algılayabilen, kararlar alabilen ve belirli hedeflere ulaşmak için eyleme geçebilen sistemleri ifade eder. 

Tek bir model komutunun (prompt) aksine, bir ajan genellikle akıl yürütme, planlama ve eyleme geçme döngüsünü izler; genellikle karmaşık görevleri tamamlamak için arama motorları, API'ler veya bellek gibi araçları entegre eder.

Bu eğitim, "ReAct" (Reasoning and Acting) adı verilen popüler bir yapay zeka ajanı mimarisine odaklanmaktadır. ReAct, dil modeline (LM) bir görev tanımıyla birlikte bir araçlar listesi sunar, ardından dil modelinin daha fazla gözlem yapmak için bu araçları çağırıp çağırmayacağına veya nihai çıktıyı üretip üretmeyeceğine karar vermesini sağlar.

Demo olarak, aşağıdaki işlemleri yapabilen basit bir havayolu müşteri hizmetleri ajanı oluşturacağız:
* Kullanıcı adına yeni seyahat rezervasyonları yapmak.
* Uçuş değişikliği ve iptali dahil olmak üzere mevcut seyahatleri değiştirmek.
* Üstesinden gelemeyeceği görevlerde bir müşteri destek bileti (ticket) açmak.

Bunu `dspy.ReAct` modülünden inşa edeceğiz.

## Bağımlılıkları Yükleme

Başlamadan önce gerekli paketleri yükleyelim:

```bash
pip install -qU dspy pydantic
```

**Önerilen:** Arka planda neler olduğunu anlamak için MLflow Tracing'i kurun.
MLflow, DSPy ile yerel olarak entegre olan, açıklanabilirlik ve deney takibi sunan bir LLMOps aracıdır. Bu eğitimde, DSPy'nin davranışını daha iyi anlamak amacıyla komutları ve optimizasyon sürecini "trace" (iz) olarak görselleştirmek için MLflow'u kullanabilirsiniz. MLflow'u `mlflow.dspy.autolog()` ile kolayca etkinleştirebilirsiniz.

## Araçları Tanımlama

Ajanın bir havayolu müşteri temsilcisi gibi davranabilmesi için bir araç listesi hazırlamamız gerekiyor:
* `fetch_flight_info` : Belirli tarihler için uçuş bilgilerini getirir.
* `pick_flight` : Belirli kriterlere göre en iyi uçuşu seçer.
* `book_flight` : Kullanıcı adına uçuş rezervasyonu yapar.
* `fetch_itinerary` : Rezerve edilmiş bir seyahatin bilgilerini getirir.
* `cancel_itinerary` : Rezerve edilmiş bir seyahati iptal eder.
* `get_user_info` : Kullanıcıların bilgilerini getirir.
* `file_ticket` : İnsan asistanlığı gerektiren durumlarda bir birikmiş iş (backlog) bileti oluşturur.

## Veri Yapısını Tanımlama

Araçları tanımlamadan önce veri yapısını tanımlamamız gerekiyor. Gerçek üretim ortamında bu, veritabanı şeması olacaktır. Demo amaçlı basitlik için veri yapısını pydantic modelleri olarak tanımlayacağız.

```python
from pydantic import BaseModel

class Date(BaseModel):
    year: int
    month: int
    day: int
    hour: int

class UserProfile(BaseModel):
    user_id: str
    name: str
    email: str

class Flight(BaseModel):
    flight_id: str
    date_time: Date
    origin: str
    destination: str
    duration: float
    price: float

class Itinerary(BaseModel):
    confirmation_number: str
    user_profile: UserProfile
    flight: Flight

class Ticket(BaseModel):
    user_request: str
    user_profile: UserProfile
```

## Sahte Veri Oluşturma

Havayolu ajanının çalışabilmesi için biraz sahte veri oluşturalım. Birkaç uçuş ve birkaç kullanıcı oluşturmalı, seyahat planları ve müşteri destek biletleri için boş sözlükler (dictionaries) başlatmalıyız.

```python
user_database = {
    "Adam": UserProfile(user_id="1", name="Adam", email="adam@gmail.com"),
    "Bob": UserProfile(user_id="2", name="Bob", email="bob@gmail.com"),
}

flight_database = {
    "DA123": Flight(
        flight_id="DA123", origin="SFO", destination="JFK", 
        date_time=Date(year=2025, month=9, day=1, hour=1), duration=3.0, price=200.0
    ),
    "DA125": Flight(
        flight_id="DA125", origin="SFO", destination="JFK", 
        date_time=Date(year=2025, month=9, day=1, hour=7), duration=9.0, price=500.0
    ),
}

itinery_database = {}
ticket_database = {}
```

## Araçları Tanımlama

Artık araçları tanımlayabiliriz. `dspy.ReAct` fonksiyonunun düzgün çalışabilmesi için her fonksiyon şunlara sahip olmalıdır:
* Aracın ne işe yaradığını tanımlayan bir "docstring". Eğer fonksiyon adı kendi kendini açıklıyorsa docstring'i boş bırakabilirsiniz.
* Dil modelinin argümanları doğru formatta üretebilmesi için gerekli olan tür ipuçları (type hint).

```python
def fetch_flight_info(date: Date, origin: str, destination: str):
    """Fetch flight information from origin to destination on the given date"""
    flights = []
    for flight_id, flight in flight_database.items():
        if (
            flight.date_time.year == date.year and
            flight.date_time.month == date.month and
            flight.date_time.day == date.day and
            flight.origin == origin and
            flight.destination == destination
        ):
            flights.append(flight)
    return flights

# (Not: Diğer tüm araçlar da benzer mantıkla Python fonksiyonları olarak yazılır.)
```

## ReAct Ajanını Oluşturma

Artık `dspy.ReAct` aracılığıyla ReAct ajanını oluşturabiliriz. Görevi, ajanın girdi ve çıktılarını tanımlamak ve erişebileceği araçları ona bildirmek için `dspy.ReAct`'e bir imza (signature) sağlamalıyız.

```python
import dspy
import os

os.environ["OPENAI_API_KEY"] = "{kendi_openai_api_anahtariniz}"
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class DSPyAirlineCustomerService(dspy.Signature):
    """You are an airline customer service agent that helps user book and manage flights. 
    You are given a list of tools to handle user request, and you should decide the right tool 
    to use in order to fulfill users' request."""
    
    user_request: str = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc=("Message that summarizes the process result, and the information users need, "
              "e.g., the confirmation_number if a new flight is booked.")
    )

agent = dspy.ReAct(
    DSPyAirlineCustomerService, 
    tools=[
        fetch_flight_info, 
        # (Diğer tanımlanan araç fonksiyonları buraya eklenir)
    ]
)
```

## Ajanı Kullanma

Ajanla etkileşime geçmek için talebi sadece `user_request` üzerinden iletmeniz yeterlidir, ardından ajan görevini yapmaya başlayacaktır.

```python
response = agent(user_request="Merhaba, benim adım Bob. Bana 1 Eylül 2025 tarihi için SFO'dan JFK'ye bir uçuş rezerve edebilir misin?")
print(response.process_result)
```

## Sonucu Yorumlama

Ajanın bir yörünge (trajectory) alanı vardır ve şunları içerir:
* Her adımda Akıl Yürütme (Reasoning/Thought).
* Her adımda dil modeli tarafından seçilen Araçlar (Tools picked).
* Her adımda dil modeli tarafından belirlenen, araç çağrısı için Argümanlar.
* Her adımda Araç çalıştırma sonuçları (Tool execution results).