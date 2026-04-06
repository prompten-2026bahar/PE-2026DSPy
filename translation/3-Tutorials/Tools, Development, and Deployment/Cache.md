# DSPy Önbelleğini Kullanma ve Özelleştirme

Bu eğitimde, DSPy’nin önbellekleme mekanizmasının tasarımını inceleyecek ve bunu etkili biçimde nasıl kullanıp özelleştirebileceğinizi göstereceğiz.

## DSPy Önbellek Yapısı

DSPy’nin önbellekleme sistemi üç ayrı katmanda tasarlanmıştır:

1. **Bellek içi önbellek**: `cachetools.LRUCache` ile uygulanır; sık kullanılan verilere hızlı erişim sağlar.
2. **Disk üzeri önbellek**: `diskcache.FanoutCache` kullanır; önbelleğe alınan öğeler için kalıcı depolama sunar.
3. **İstem önbelleği (sunucu tarafı önbellek)**: Bu katman LLM hizmet sağlayıcısı (ör. OpenAI, Anthropic) tarafından yönetilir.

DSPy, sunucu tarafındaki istem önbelleğini doğrudan kontrol etmez; ancak kullanıcıların bellek içi ve disk üzeri önbellekleri kendi ihtiyaçlarına göre etkinleştirmesine, devre dışı bırakmasına ve özelleştirmesine esneklik sağlar.

## DSPy Önbelleğini Kullanma

Varsayılan olarak hem bellek içi hem de disk üzeri önbellekleme DSPy’de otomatik olarak etkindir. Önbelleği kullanmaya başlamak için özel bir işlem yapmanız gerekmez. Bir önbellek isabeti olduğunda, modül çağrısının çalışma süresinde belirgin bir azalma görürsünüz. Ayrıca kullanım takibi etkinse, önbellekten gelen bir çağrı için kullanım metrikleri `None` olur.

Aşağıdaki örneği ele alalım:

```python
import dspy
import os
import time

os.environ["OPENAI_API_KEY"] = "{your_openai_key}"

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), track_usage=True)

predict = dspy.Predict("question->answer")

start = time.time()
result1 = predict(question="Who is the GOAT of basketball?")
print(f"Time elapse: {time.time() - start: 2f}\n\nTotal usage: {result1.get_lm_usage()}")

start = time.time()
result2 = predict(question="Who is the GOAT of basketball?")
print(f"Time elapse: {time.time() - start: 2f}\n\nTotal usage: {result2.get_lm_usage()}")
```

Örnek bir çıktı şöyledir:

```
Time elapse:  4.384113
Total usage: {'openai/gpt-4o-mini': {'completion_tokens': 97, 'prompt_tokens': 144, 'total_tokens': 241, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0, 'text_tokens': None}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0, 'text_tokens': None, 'image_tokens': None}}}

Time elapse:  0.000529
Total usage: {}
```

## Sağlayıcı Taraflı İstem Önbelleğini Kullanma

DSPy’nin yerleşik önbellekleme mekanizmasına ek olarak, Anthropic ve OpenAI gibi LLM sağlayıcılarının sunduğu sağlayıcı taraflı istem önbelleklemeden de yararlanabilirsiniz. Bu özellik özellikle `dspy.ReAct()` gibi benzer istemleri tekrar tekrar gönderen modüllerle çalışırken yararlıdır; çünkü istem öneklerini sağlayıcının sunucularında önbelleğe alarak hem gecikmeyi hem de maliyeti azaltır.

İstem önbelleğini, `dspy.LM()` içine `cache_control_injection_points` parametresini geçirerek etkinleştirebilirsiniz. Bu özellik Anthropic ve OpenAI gibi desteklenen sağlayıcılarla çalışır. Ayrıntılar için [LiteLLM prompt caching documentation](https://docs.litellm.ai/docs/tutorials/prompt_caching#configuration) sayfasına bakın.

```python
import dspy
import os

os.environ["ANTHROPIC_API_KEY"] = "{your_anthropic_key}"
lm = dspy.LM(
    "anthropic/claude-sonnet-4-5-20250929",
    cache_control_injection_points=[
        {
            "location": "message",
            "role": "system",
        }
    ],
)
dspy.configure(lm=lm)

# Herhangi bir DSPy modülüyle kullanın
predict = dspy.Predict("question->answer")
result = predict(question="What is the capital of France?")
```

Bu özellikle şu durumlarda faydalıdır:

- Aynı talimatlarla `dspy.ReAct()` kullanırken
- Sabit kalan uzun sistem istemleriyle çalışırken
- Benzer bağlamla birden fazla istek yaparken

## DSPy Önbelleğini Devre Dışı Bırakma / Etkinleştirme

Bazı durumlarda önbelleği tamamen ya da yalnızca bellek içi veya disk üzeri önbellek için seçici biçimde kapatmanız gerekebilir. Örneğin:

- Aynı LM isteği için farklı yanıtlar istiyorsunuzdur.
- Diske yazma izniniz yoktur ve disk üzeri önbelleği kapatmanız gerekir.
- Bellek kaynaklarınız sınırlıdır ve bellek içi önbelleği kapatmak istersiniz.

DSPy bu amaçla `dspy.configure_cache()` yardımcı işlevini sağlar. Her önbellek türünün etkin/devre dışı durumunu ilgili bayraklarla kontrol edebilirsiniz:

```python
dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)
```

Ek olarak, bellek içi ve disk üzeri önbelleklerin kapasitesini de yönetebilirsiniz:

```python
dspy.configure_cache(
    enable_disk_cache=True,
    enable_memory_cache=True,
    disk_size_limit_bytes=YOUR_DESIRED_VALUE,
    memory_max_entries=YOUR_DESIRED_VALUE,
)
```

Lütfen `disk_size_limit_bytes` parametresinin disk üzeri önbellek için bayt cinsinden azami boyutu, `memory_max_entries` parametresinin ise bellek içi önbellek için azami giriş sayısını tanımladığını unutmayın.

## Önbelleği Anlama ve Özelleştirme

Bazı özel durumlarda, örneğin önbellek anahtarlarının nasıl üretileceği üzerinde daha ayrıntılı denetime sahip olmak isteyebilirsiniz. Varsayılan olarak önbellek anahtarı, `api_key` gibi kimlik bilgileri hariç, `litellm`’e gönderilen tüm istek argümanlarının bir karmasından türetilir.

Özel bir önbellek oluşturmak için `dspy.clients.Cache` sınıfını alt sınıflandırmalı ve ilgili metotları ezmelisiniz:

```python
class CustomCache(dspy.clients.Cache):
    def __init__(self, **kwargs):
        {write your own constructor}

    def cache_key(self, request: dict[str, Any], ignored_args_for_cache_key: Optional[list[str]] = None) -> str:
        {write your logic of computing cache key}

    def get(self, request: dict[str, Any], ignored_args_for_cache_key: Optional[list[str]] = None) -> Any:
        {write your cache read logic}

    def put(
        self,
        request: dict[str, Any],
        value: Any,
        ignored_args_for_cache_key: Optional[list[str]] = None,
        enable_memory_cache: bool = True,
    ) -> None:
        {write your cache write logic}
```

DSPy’nin geri kalanıyla sorunsuz entegrasyon için, özel önbelleğinizi temel sınıfla aynı metot imzalarını kullanarak uygulamanız önerilir; en azından önbellek okuma/yazma işlemleri sırasında çalışma zamanı hatalarını önlemek için metot tanımlarınıza `**kwargs` ekleyin.

Özel önbellek sınıfınızı tanımladıktan sonra DSPy’ye bunu kullanmasını söyleyebilirsiniz:

```python
dspy.cache = CustomCache()
```

Bunu pratik bir örnekle gösterelim. Diyelim ki önbellek anahtarının yalnızca istek mesajı içeriğine bağlı olmasını, çağrılan belirli LM gibi diğer parametrelerin göz ardı edilmesini istiyoruz. O zaman aşağıdaki gibi özel bir önbellek oluşturabiliriz:

```python
class CustomCache(dspy.clients.Cache):

    def cache_key(self, request: dict[str, Any], ignored_args_for_cache_key: Optional[list[str]] = None) -> str:
        messages = request.get("messages", [])
        return sha256(orjson.dumps(messages, option=orjson.OPT_SORT_KEYS)).hexdigest()

dspy.cache = CustomCache(enable_disk_cache=True, enable_memory_cache=True, disk_cache_dir=dspy.clients.DISK_CACHE_DIR)
```

Karşılaştırma için, aşağıdaki kodu özel önbellek olmadan çalıştırmayı düşünün:

```python
import dspy
import os
import time

os.environ["OPENAI_API_KEY"] = "{your_openai_key}"

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

predict = dspy.Predict("question->answer")

start = time.time()
result1 = predict(question="Who is the GOAT of soccer?")
print(f"Time elapse: {time.time() - start: 2f}")

start = time.time()
with dspy.context(lm=dspy.LM("openai/gpt-4.1-mini")):
    result2 = predict(question="Who is the GOAT of soccer?")
print(f"Time elapse: {time.time() - start: 2f}")
```

Geçen süre, ikinci çağrıda önbellek isabeti olmadığını gösterecektir. Ancak özel önbellek kullanıldığında:

```python
import dspy
import os
import time
from typing import Dict, Any, Optional
import orjson
from hashlib import sha256

os.environ["OPENAI_API_KEY"] = "{your_openai_key}"

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class CustomCache(dspy.clients.Cache):

    def cache_key(self, request: dict[str, Any], ignored_args_for_cache_key: Optional[list[str]] = None) -> str:
        messages = request.get("messages", [])
        return sha256(orjson.dumps(messages, option=orjson.OPT_SORT_KEYS)).hexdigest()

dspy.cache = CustomCache(enable_disk_cache=True, enable_memory_cache=True, disk_cache_dir=dspy.clients.DISK_CACHE_DIR)

predict = dspy.Predict("question->answer")

start = time.time()
result1 = predict(question="Who is the GOAT of volleyball?")
print(f"Time elapse: {time.time() - start: 2f}")

start = time.time()
with dspy.context(lm=dspy.LM("openai/gpt-4.1-mini")):
    result2 = predict(question="Who is the GOAT of volleyball?")
print(f"Time elapse: {time.time() - start: 2f}")
```

İkinci çağrıda önbellek isabeti olduğunu gözlemleyeceksiniz; bu da özel önbellek anahtarı mantığının etkisini gösterir.
