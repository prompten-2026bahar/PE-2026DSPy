# Async DSPy Programlama

DSPy, daha verimli ve ölçeklenebilir uygulamalar geliştirmenize olanak tanıyan yerel asenkron programlama desteği sunar. Bu kılavuz, hem yerleşik modülleri hem de özel uygulamaları kapsayacak şekilde DSPy’de async yeteneklerden nasıl yararlanacağınızı adım adım açıklar.

## DSPy’de Neden Async Kullanılmalı?

DSPy’de asenkron programlama çeşitli avantajlar sunar:
- Eşzamanlı işlemler sayesinde daha iyi performans
- Kaynakların daha verimli kullanımı
- G/Ç ağırlıklı işlemlerde daha az bekleme süresi
- Birden fazla isteği işlemek için daha yüksek ölçeklenebilirlik

## Sync mi Async mi Kullanmalıyım?

DSPy’de senkron ve asenkron programlama arasında seçim yapmak, özel kullanım senaryonuza bağlıdır.
Doğru seçimi yapmanıza yardımcı olacak kısa bir rehber aşağıdadır:

Senkron Programlamayı Şu Durumlarda Kullanın

- Yeni fikirleri keşfediyor veya prototip geliştiriyorsanız
- Araştırma ya da deney yapıyorsanız
- Küçük veya orta ölçekli uygulamalar geliştiriyorsanız
- Daha basit ve daha doğrudan bir kod yapısına ihtiyaç duyuyorsanız
- Daha kolay hata ayıklama ve hata takibi istiyorsanız

Asenkron Programlamayı Şu Durumlarda Kullanın:

- Yüksek aktarım hacimli bir servis geliştiriyorsanız (yüksek QPS)
- Yalnızca async işlemleri destekleyen araçlarla çalışıyorsanız
- Birden fazla eşzamanlı isteği verimli biçimde yönetmeniz gerekiyorsa
- Yüksek ölçeklenebilirlik gerektiren üretim servisleri geliştiriyorsanız

### Önemli Hususlar

Async programlama performans avantajları sunsa da bazı ödünleşimleri de beraberinde getirir:

- Daha karmaşık hata yönetimi ve hata ayıklama
- Takibi zor, ince hatalar oluşabilmesi
- Daha karmaşık kod yapısı
- ipython (Colab, Jupyter lab, Databricks not defterleri, ...) ile normal Python çalışma zamanı arasında farklı kod yapısı

Çoğu geliştirme senaryosunda senkron programlamayla başlamanızı ve yalnızca async avantajlarına gerçekten ihtiyaç duyduğunuzda asenkrona geçmenizi öneririz. Bu yaklaşım, async programlamanın ek karmaşıklığıyla uğraşmadan önce uygulamanızın temel mantığına odaklanmanızı sağlar.

## Yerleşik Modülleri Asenkron Kullanma

DSPy’nin çoğu yerleşik modülü, `acall()` metodu üzerinden asenkron çalışmayı destekler. Bu metod, senkron `__call__` metoduyla aynı arayüzü korur ancak asenkron olarak çalışır.

Aşağıda `dspy.Predict` kullanan temel bir örnek verilmiştir:

```python
import dspy
import asyncio
import os

os.environ["OPENAI_API_KEY"] = "your_api_key"

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
predict = dspy.Predict("question->answer")

async def main():
    # Asenkron yürütme için acall() kullan
    output = await predict.acall(question="why did a chicken cross the kitchen?")
    print(output)


asyncio.run(main())
```

### Async Araçlarla Çalışma

DSPy’nin `Tool` sınıfı, async fonksiyonlarla sorunsuz biçimde entegre olur. Bir async fonksiyonu `dspy.Tool` içine verdiğinizde, bunu `acall()` ile çalıştırabilirsiniz. Bu özellikle G/Ç ağırlıklı işlemler veya harici servislerle çalışırken kullanışlıdır.

```python
import asyncio
import dspy
import os

os.environ["OPENAI_API_KEY"] = "your_api_key"

async def foo(x):
    # Bir async işlemi taklit et
    await asyncio.sleep(0.1)
    print(f"I get: {x}")

# Async fonksiyondan bir araç oluştur
tool = dspy.Tool(foo)

async def main():
    # Aracı asenkron olarak çalıştır
    await tool.acall(x=2)

asyncio.run(main())
```

#### Async Araçları Senkron Bağlamlarda Kullanma

Bir async aracı senkron kod içinden çağırmanız gerekiyorsa, otomatik async-to-sync dönüşümünü etkinleştirebilirsiniz:

```python
import dspy

async def async_tool(x: int) -> int:
    """Bir sayıyı ikiyle çarpan async araç."""
    await asyncio.sleep(0.1)
    return x * 2

tool = dspy.Tool(async_tool)

# Seçenek 1: Geçici dönüşüm için context manager kullan
with dspy.context(allow_tool_async_sync_conversion=True):
    result = tool(x=5)  # Senkron bağlamda çalışır
    print(result)  # 10

# Seçenek 2: Global olarak yapılandır
dspy.configure(allow_tool_async_sync_conversion=True)
result = tool(x=5)  # Artık her yerde çalışır
print(result)  # 10
```

Async araçlar hakkında daha fazla ayrıntı için [Araçlar dokümantasyonuna](../../learn/programming/tools.md#async-tools) bakın.

Not: `dspy.ReAct` ile araç kullanırken, ReAct örneği üzerinde `acall()` çağırmak tüm araçları otomatik olarak `acall()` metodlarıyla asenkron şekilde çalıştırır.

## Özel Async DSPy Modülleri Oluşturma

Kendi async DSPy modülünüzü oluşturmak için `forward()` yerine `aforward()` metodunu uygulayın. Bu metod, modülünüzün async mantığını içermelidir. Aşağıda iki async işlemi zincirleyen özel bir modül örneği verilmiştir:

```python
import dspy
import asyncio
import os

os.environ["OPENAI_API_KEY"] = "your_api_key"
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class MyModule(dspy.Module):
    def __init__(self):
        self.predict1 = dspy.ChainOfThought("question->answer")
        self.predict2 = dspy.ChainOfThought("answer->simplified_answer")

    async def aforward(self, question, **kwargs):
        # Tahminleri sıralı ama asenkron biçimde çalıştır
        answer = await self.predict1.acall(question=question)
        return await self.predict2.acall(answer=answer)


async def main():
    mod = MyModule()
    result = await mod.acall(question="Why did a chicken cross the kitchen?")
    print(result)


asyncio.run(main())
```
