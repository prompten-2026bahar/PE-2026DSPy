# /translation/rlm.md

## dspy.RLM

**RLM (Recursive Language Model / Özyineli Dil Modeli)**, YZ modellerinin (LLM) korumalı (sandboxed) bir Python REPL'i (Okuma-Değerlendirme-Yazdırma Döngüsü) aracılığıyla büyük bağlamları programatik olarak keşfetmesini sağlayan bir DSPy modülüdür. Devasa bağlamları doğrudan istemin (prompt) içine beslemek yerine, RLM bağlamı LLM'nin kod yürütme (code execution) ve özyineli alt-LLM çağrılarıyla incelediği harici bir veri olarak ele alır.

Bu, "Recursive Language Models" (Zhang, Kraska, Khattab, 2025) makalesinde açıklanan yaklaşımı uygular.

### RLM Ne Zaman Kullanılmalı (When to Use RLM)

Bağlamlar büyüdükçe LLM performansı düşer — bu olguya **bağlam çürümesi (context rot)** denir. RLM'ler, **değişken alanını (variable space)** (REPL'de saklanan bilgi) **token alanından (token space)** (LLM'nin fiilen işlediği kısım) ayırarak bu sorunu çözer. LLM yalnızca ihtiyaç duyduğu bağlamı, ihtiyaç duyduğu anda dinamik olarak yükler.

Şu durumlarda RLM kullanın:
* Bağlamınız LLM'nin bağlam penceresine (context window) etkili bir şekilde sığmayacak kadar **çok büyükse**.
* Görev, **programatik keşiften** (arama, filtreleme, toplama, parçalama) fayda sağlıyorsa.
* Problemi **nasıl parçalara ayıracağınıza** sizin değil, LLM'nin karar vermesine ihtiyacınız varsa.

### Temel Kullanım (Basic Usage)

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-5"))

# Bir RLM modülü oluşturun
rlm = dspy.RLM("context, query -> answer")

# Diğer herhangi bir modül gibi çağırın
result = rlm(
    context="...çok uzun bir belge veya veri...",
    query="Bahsedilen toplam gelir nedir?"
)
print(result.answer)
```

### Deno Kurulumu (Deno Installation)

RLM, güvenli Python yürütmesi için yerel bir WASM korumalı alanı (sandbox) oluşturmak üzere **Deno** ve **Pyodide** teknolojilerine güvenir.

macOS ve Linux üzerinde Deno'yu şu komutla kurabilirsiniz: `curl -fsSL https://deno.land/install.sh | sh`. Daha fazla detay için Deno Kurulum Belgelerine bakın. Komut, kabuk profilinize (shell profile) ekleme yapmak istediğinde bu istemi kabul ettiğinizden emin olun.

Deno'yu kurduktan sonra, **kabuğunuzu (shell) yeniden başlattığınızdan emin olun.**
Ardından `dspy.RLM`'yi çalıştırabilirsiniz.

Kullanıcılar, DSPy tarafından Deno önbelleğinin bulunamamasıyla ilgili sorunlar bildirmiştir. Bu sorunları aktif olarak araştırıyoruz ve geri bildirimleriniz bizim için çok değerlidir.

Ayrıca harici bir sandbox (korumalı alan) sağlayıcısıyla da çalışabilirsiniz. Harici sandbox sağlayıcılarının kullanımına dair bir örnek oluşturmak üzerinde hâlâ çalışıyoruz.

### Nasıl Çalışır (How It Works)

RLM yinelemeli bir REPL döngüsünde (iterative REPL loop) çalışır:

1.  LLM bağlam hakkında **meta veriler** alır (tip, uzunluk, önizleme) ancak tam bağlamı (full context) almaz.
2.  LLM verileri keşfetmek için **Python kodu** yazar (örnekleri yazdırma, arama, filtreleme).
3.  Kod **korumalı bir yorumlayıcıda (sandboxed interpreter)** çalışır ve LLM çıktıyı görür.
4.  LLM, kod parçacıkları üzerinde anlamsal analiz (semantic analysis) yapmak amacıyla **alt-LLM çağrıları (sub-LLM calls)** çalıştırmak için `llm_query(prompt)` fonksiyonunu çağırabilir.
5.  İşlem bittiğinde, LLM nihai cevabı döndürmek için `SUBMIT(output)` fonksiyonunu çağırır.

#### LLM'nin Gördükleri (Adım Adım İzleme):

**ADIM 1: İLK META VERİ (TAM BAĞLAMA DOĞRUDAN ERİŞİM YOK)**
```python
# Adım 1: Veriye göz at
print(context[:2000])
```
*LLM'ye gösterilen çıktı:*
[Belgenin ilk 2000 karakterinin önizlemesi]

**ADIM 2: BAĞLAMI KEŞFETMEK İÇİN KOD YAZMA**
```python
# Adım 2: İlgili bölümleri ara
import re
matches = re.findall(r'revenue.*?\$[\d,]+', context, re.IGNORECASE)
print(matches)
```
*LLM'ye gösterilen çıktı:*
['Revenue in Q4: $5,000,000', 'Total revenue: $20,000,000']

**ADIM 3: ALT-LLM ÇAĞRILARINI TETİKLEME**
```python
# Adım 3: Anlamsal çıkarım için alt-LLM kullan
result = llm_query(f"Extract the total revenue from: {matches[1]}")
print(result)
```
*LLM'ye gösterilen çıktı:*
$20,000,000

**ADIM 4: NİHAİ CEVABI GÖNDERME**
```python
# Adım 4: Nihai cevabı döndür
SUBMIT(result)
```
*Kullanıcıya gösterilen çıktı:*
$20,000,000

### Kurucu Parametreleri (Constructor Parameters)

| Parametre (Name) | Tip (Type) | Varsayılan (Default) | Açıklama (Description) |
| :--- | :--- | :--- | :--- |
| `signature` | `str \| Signature` | **gerekli** | Girdileri ve çıktıları tanımlar (örn. `"context, query -> answer"`). |
| `max_iterations` | `int` | `20` | Yedek (fallback) çıkarımdan önceki maksimum REPL etkileşim döngüsü. |
| `max_llm_calls` | `int` | `50` | Her yürütme başına maksimum `llm_query`/`llm_query_batched` çağrısı. |
| `max_output_chars` | `int` | `10_000` | REPL çıktısından dahil edilecek maksimum karakter sayısı. |
| `verbose` | `bool` | `False` | Detaylı yürütme bilgilerini günlüğe kaydeder (log). |
| `tools` | `list[Union[Callable, dspy.Tool]]` | `None` | Yorumlayıcı kodundan çağrılabilecek ek araç fonksiyonları. |
| `sub_lm` | `dspy.LM` | `None` | Alt sorgular (sub-queries) için LM. Varsayılanı `dspy.settings.lm`'dir. Burada daha ucuz bir model kullanın. |
| `interpreter` | `CodeInterpreter` | `None` | Özel yorumlayıcı (Custom interpreter). Varsayılanı `PythonInterpreter`'dır (Deno/Pyodide WASM). |

### Yerleşik Araçlar (Built-in Tools)

REPL içerisinde LLM şunlara erişebilir:

| Araç (Tool) | Açıklama (Description) |
| :--- | :--- |
| `llm_query(prompt)` | Anlamsal analiz için bir alt-LLM'i sorgular (~500K karakter kapasitesi). |
| `llm_query_batched(prompts)` | Eşzamanlı olarak birden fazla istemi sorgular (toplu işlemler için daha hızlıdır). |
| `print()` | Çıktıyı yazdırır (sonuçları görmek için gereklidir). |
| `SUBMIT(...)` | Nihai çıktıyı gönderir ve yürütmeyi sonlandırır. |
| Standart Kütüphane | `re`, `json`, `collections`, `math`, vb. |

### Örnekler (Examples)

#### Uzun Belge Soru-Cevap (Long Document Q&A)
```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-5"))
rlm = dspy.RLM("document, question -> answer", max_iterations=10)

with open("large_report.txt") as f:
    document = f.read()  # 500.000+ karakter

result = rlm(
    document=document,
    question="What were the key findings from Q3?"
)
print(result.answer)
```

#### Daha Ucuz Bir Alt-LLM Kullanımı (Using a Cheaper Sub-LM)
```python
import dspy

main_lm = dspy.LM("openai/gpt-5")
cheap_lm = dspy.LM("openai/gpt-5-nano")
dspy.configure(lm=main_lm)

# Kök LM (gpt-5) stratejiye karar verir; alt-LM (gpt-5-nano) çıkarımı halleder
rlm = dspy.RLM("data, query -> summary", sub_lm=cheap_lm)
```

#### Çoklu Tiplendirilmiş Çıktılar (Multiple Typed Outputs)
```python
rlm = dspy.RLM("logs -> error_count: int, critical_errors: list[str]")
result = rlm(logs=server_logs)

print(f"Found {result.error_count} errors")
print(f"Critical: {result.critical_errors}")
```

#### Özel Araçlar (Custom Tools)
```python
def fetch_metadata(doc_id: str) -> str:
    '''Bir belge kimliği (document ID) için meta veri getirir.'''
    return database.get_metadata(doc_id)

rlm = dspy.RLM(
    "documents, query -> answer",
    tools=[fetch_metadata]
)
```

#### Asenkron Yürütme (Async Execution)
```python
import asyncio

rlm = dspy.RLM("context, query -> answer")

async def process():
    result = await rlm.aforward(context=data, query="Summarize this")
    return result.answer

answer = asyncio.run(process())
```

#### Yörüngeyi/Geçmişi İnceleme (Inspecting the Trajectory)
```python
result = rlm(context=data, query="Find the magic number")

# LLM'nin hangi kodu çalıştırdığını görün
for step in result.trajectory:
    print(f"Code:\n{step['code']}")
    print(f"Output:\n{step['output']}\n")
```

### Çıktı (Output)

RLM, aşağıdakileri içeren bir `Prediction` döndürür:
* İmzanızdaki (signature) **çıktı alanları (Output fields)** (örn. `result.answer`).
* **`trajectory`**: Her adım için `reasoning` (akıl yürütme), `code` (kod) ve `output` (çıktı) içeren sözlüklerin listesi.
* **`final_reasoning`**: LLM'nin son adımdaki akıl yürütmesi.

### Notlar (Notes)

**Deneysel (Experimental)**
RLM deneysel olarak işaretlenmiştir. API gelecekteki sürümlerde değişebilir.

**İş Parçacığı Güvenliği (Thread Safety)**
Özel bir yorumlayıcı (custom interpreter) kullanıldığında RLM örnekleri iş parçacığı açısından güvenli (thread-safe) değildir. Eşzamanlı kullanım için ayrı örnekler oluşturun veya her `forward()` çağrısı için yeni bir örnek oluşturan varsayılan `PythonInterpreter`'ı kullanın.

**Yorumlayıcı Gereksinimleri (Interpreter Requirements)**
Varsayılan `PythonInterpreter`, Pyodide WASM sandbox'ı için **Deno**'nun kurulu olmasını gerektirir.

---

## API Reference (API Referansı)

### `dspy.RLM`

```python
dspy.RLM(signature: type[Signature] | str, max_iterations: int = 20, max_llm_calls: int = 50, max_output_chars: int = 10000, verbose: bool = False, tools: list[Callable] | None = None, sub_lm: dspy.LM | None = None, interpreter: CodeInterpreter | None = None)
```

**Kullanılan Yapılar (Bases):** `Module`

Özyineli Dil Modeli (Recursive Language Model) modülü.

LLM'nin programatik olarak büyük bağlamları kod yürütme (code execution) yoluyla keşfetmesine izin vermek için korumalı bir REPL kullanır. LLM, verileri incelemek, anlamsal analiz için alt-LLM'leri çağırmak ve yanıtları yinelemeli olarak oluşturmak için Python kodu yazar.

Varsayılan yorumlayıcı `PythonInterpreter`'dır (Deno/Pyodide/WASM), ancak herhangi bir `CodeInterpreter` uygulaması (örn. `MockInterpreter` sağlayabilir veya E2B ya da Modal kullanarak özel bir tane yazabilirsiniz) sağlayabilirsiniz.

**Not:** Özel bir yorumlayıcı kullanıldığında RLM örnekleri iş parçacığı güvenli (thread-safe) değildir. Eşzamanlı kullanım için ayrı RLM örnekleri oluşturun veya her `forward()` çağrısında yeni bir örnek oluşturan varsayılan `PythonInterpreter`'ı kullanın.

**Örnekler (Examples):**

```python
# Temel kullanım
rlm = dspy.RLM("context, query -> output", max_iterations=10)
result = rlm(context="...very long text...", query="What is the magic number?")
print(result.output)
```

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `signature` | `type[Signature] \| str` | Girdileri ve çıktıları tanımlar. "context, query -> answer" gibi bir string veya bir `Signature` sınıfı olabilir. | **Gerekli (required)** |
| `max_iterations` | `int` | Maksimum REPL etkileşim yinelemeleri. | `20` |
| `max_llm_calls` | `int` | Yürütme başına maksimum alt-LLM çağrısı (`llm_query`/`llm_query_batched`). | `50` |
| `max_output_chars` | `int` | REPL çıktısından dahil edilecek maksimum karakter sayısı. | `10000` |
| `verbose` | `bool` | Detaylı yürütme bilgilerinin günlüğe kaydedilip kaydedilmeyeceği. | `False` |
| `tools` | `list[Callable] \| None` | Yorumlayıcı kodundan çağrılabilecek araç fonksiyonlarının veya `dspy.Tool` nesnelerinin listesi. Yerleşik araçlar: `llm_query(prompt)`, `llm_query_batched(prompts)`. | `None` |
| `sub_lm` | `LM \| None` | `llm_query`/`llm_query_batched` için LM. Varsayılanı `dspy.settings.lm`'dir. Alt sorgular için farklı (örn. daha ucuz) bir model kullanılmasına olanak tanır. | `None` |
| `interpreter` | `CodeInterpreter \| None` | Kullanılacak `CodeInterpreter` uygulaması. Varsayılanı `PythonInterpreter`'dır. | `None` |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/predict/rlm.py`*

---

## Öznitelikler (Attributes)

### `tools`
```python
tools: dict[str, Tool] # property
```
Kullanıcı tarafından sağlanan araçlar (dahili `llm_query`/`llm_query_batched` fonksiyonlarını içermez).

---

## Functions (Fonksiyonlar)

### `__call__`

```python
__call__(*args, **kwargs) -> Prediction
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/module.py`*

### `forward`

```python
forward(**input_args) -> Prediction
```

Verilen girdilerden çıktılar üretmek için RLM'yi yürütür.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `**input_args` | - | İmzanın (signature) girdi alanlarıyla eşleşen girdi değerleri. | `{}` |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `Prediction` | İmzadan gelen çıktı alanı/alanları ve hata ayıklama (debugging) için 'trajectory' (yörünge) içeren Prediction nesnesi. |

**Hatalar (Raises):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `ValueError` | Gerekli girdi alanları eksikse fırlatılır. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/predict/rlm.py`*

### `aforward`

```python
aforward(**input_args) -> Prediction async
```

`forward()` metodunun asenkron (async) versiyonudur. Çıktılar üretmek için RLM'yi yürütür.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `**input_args` | - | İmzanın girdi alanlarıyla eşleşen girdi değerleri. | `{}` |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `Prediction` | İmzadan gelen çıktı alanı/alanları ve hata ayıklama için 'trajectory' içeren Prediction nesnesi. |

**Hatalar (Raises):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `ValueError` | Gerekli girdi alanları eksikse fırlatılır. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/predict/rlm.py`*

### `batch`

```python
batch(examples: list[Example], num_threads: int | None = None, max_errors: int | None = None, return_failed_examples: bool = False, provide_traceback: bool | None = None, disable_progress_bar: bool = False, timeout: int = 120, straggler_limit: int = 3) -> list[Example] | tuple[list[Example], list[Example], list[Exception]]
```

`Parallel` modülünü kullanarak bir `dspy.Example` örnekleri listesini paralel olarak işler.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `examples` | `list[Example]` | İşlenecek `dspy.Example` örneklerinin listesi. | **Gerekli (required)** |
| `num_threads` | `int \| None` | Paralel işleme için kullanılacak iş parçacığı sayısı. | `None` |
| `max_errors` | `int \| None` | Yürütmeyi durdurmadan önce izin verilen maksimum hata sayısı. `None` ise `dspy.settings.max_errors` değerini miras alır. | `None` |
| `return_failed_examples` | `bool` | Başarısız örneklerin ve istisnaların döndürülüp döndürülmeyeceği. | `False` |
| `provide_traceback` | `bool \| None` | Hata günlüklerine izleme (traceback) bilgisinin dahil edilip edilmeyeceği. | `None` |
| `disable_progress_bar` | `bool` | İlerleme çubuğunun gösterilip gösterilmeyeceği. | `False` |
| `timeout` | `int` | Geride kalan (straggler) bir görevin yeniden gönderilmesinden önceki saniye süresi. Devre dışı bırakmak için 0 yapın. | `120` |
| `straggler_limit` | `int` | Yalnızca bu kadar veya daha az görev kaldığında geride kalanları kontrol eder. | `3` |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `list[Example] \| tuple[list[Example], list[Example], list[Exception]]` | Sonuçların listesi ve isteğe bağlı olarak başarısız örnekler ile istisnalar. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/module.py`*

### `deepcopy`

Modülün derin kopyasını (deep copy) oluşturur.
Bu, varsayılan Python `deepcopy` metodunun özelleştirilmiş halidir; yalnızca `self.parameters()` kısmını derin kopyalar, diğer öznitelikler için yüzeysel kopya (shallow copy) yapar.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/base_module.py`*

### `dump_state`

```python
dump_state(json_mode=True)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/base_module.py`*

### `get_lm`

Bu modülün tahmincileri (predictors) tarafından kullanılan dil modelini alır.
Tüm tahminciler aynı LM'yi kullanıyorsa dil modelini döndürür. Birden fazla farklı LM kullanılıyorsa hata verir.

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| - | Bu modülün tahmincileri tarafından kullanılan dil modeli örneği (instance). |

**Hatalar (Raises):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `ValueError` | Bu modüldeki tahminciler tarafından birden fazla farklı dil modeli kullanılıyorsa. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/module.py`*

### `load`

```python
load(path, allow_pickle=False, allow_unsafe_lm_state=False)
```

Kaydedilmiş modülü yükler. Sadece mevcut bir programın durumunu değil, tüm programı yüklemek istiyorsanız `dspy.load` fonksiyonuna da bakabilirsiniz.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `path` | `str` | `.json` veya `.pkl` dosyası olması gereken kaydedilmiş durum dosyasının yolu. | **Gerekli (required)** |
| `allow_pickle` | `bool` | `True` ise, rastgele kod çalıştırabilen `.pkl` dosyalarının yüklenmesine izin verir. Bu tehlikelidir ve yalnızca dosyanın kaynağından eminseniz ve güvenilir bir ortamdaysanız kullanılmalıdır. | `False` |
| `allow_unsafe_lm_state` | `bool` | `True` ise, yüklenen durumdan güvenli olmayan LM uç nokta anahtarlarını (örn. `api_base`, `base_url` ve `model_list`) korur. Yalnızca güvenilir dosyalar için etkinleştirin. | `False` |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/base_module.py`*

### `load_state`

```python
load_state(state, *, allow_unsafe_lm_state=False)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/base_module.py`*

### `named_parameters`

PyTorch'un aksine, (yinelenmeyen) parametre listelerini de işler.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/base_module.py`*

### `named_predictors`

Bu modüldeki tüm isimlendirilmiş `Predict` modüllerini döndürür.
Tüm parametreleri yineler ve `dspy.Predict` örneği olanları isimleriyle birlikte döndürür.

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `list[tuple[str, Predict]]` | İsmin öznitelik yolu (attribute path) ve tahmincinin `Predict` örneği olduğu `(isim, tahminci)` ikililerinin listesi. |

**Örnekler (Examples):**

```python
>>> import dspy
>>> class MyProgram(dspy.Module):
...     def __init__(self):
...         super().__init__()
...         self.qa = dspy.Predict("question -> answer")
...         self.summarize = dspy.Predict("text -> summary")
...
>>> program = MyProgram()
>>> for name, p in program.named_predictors():
...     print(name)
qa
summarize
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/module.py`*

### `parameters`

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/base_module.py`*

### `predictors`

Bu modüldeki tüm `Predict` modüllerini döndürür.

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `list[Predict]` | Bu modüldeki tüm `Predict` örneklerinin listesi. |

**Örnekler (Examples):**

```python
>>> import dspy
>>> class MyProgram(dspy.Module):
...     def __init__(self):
...         super().__init__()
...         self.qa = dspy.Predict("question -> answer")
...
>>> program = MyProgram()
>>> len(program.predictors())
1
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/module.py`*

### `reset_copy`

Modülü derin kopyalar ve tüm parametreleri sıfırlar.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/base_module.py`*

### `save`

```python
save(path, save_program=False, modules_to_serialize=None)
```

Modülü kaydeder.
Modülü bir dizine veya dosyaya kaydedin. İki mod vardır:
- `save_program=False`: Dosya uzantısının değerine bağlı olarak modülün yalnızca durumunu (state) bir json veya pickle dosyasına kaydedin.
- `save_program=True`: Modelin hem durumunu hem de mimarisini içeren tüm modülü cloudpickle aracılığıyla bir dizine kaydedin.

`save_program=True` ve `modules_to_serialize` sağlanmışsa, bu modülleri cloudpickle'ın `register_pickle_by_value` özelliği ile serileştirme için kaydeder. Bu, cloudpickle'ın modülü referans yerine değere göre serileştirmesini sağlayarak modülün kaydedilen programla birlikte tamamen korunmasını güvence altına alır. Bu, programınızla birlikte serileştirilmesi gereken özel modülleriniz olduğunda yararlıdır. `None` ise, serileştirme için hiçbir modül kaydedilmez.

Ayrıca yüklenen modelin kritik bağımlılıklarda veya DSPy sürümünde bir sürüm uyumsuzluğu olup olmadığını kontrol edebilmesi için bağımlılık sürümlerini de kaydederiz.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `path` | `str` | `save_program=False` olduğunda bir `.json` veya `.pkl` dosyası, `save_program=True` olduğunda ise bir dizin (directory) olması gereken kaydedilmiş durum dosyasının yolu. | **Gerekli (required)** |
| `save_program` | `bool` | `True` ise tüm modülü cloudpickle aracılığıyla bir dizine kaydeder, aksi takdirde yalnızca durumu kaydeder. | `False` |
| `modules_to_serialize` | `list` | Cloudpickle'ın `register_pickle_by_value` özelliği ile serileştirilecek modüllerin listesi. `None` ise, serileştirme için hiçbir modül kaydedilmez. | `None` |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/base_module.py`*

### `set_lm`

Bu modüldeki tüm tahminciler için dil modelini ayarlar.
Bu metot, bu modül içinde bulunan tüm `Predict` örnekleri için dil modelini yinelemeli (recursively) olarak ayarlar.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `lm` | - | Tüm tahminciler için kullanılacak dil modeli örneği. | **Gerekli (required)** |

**Örnekler (Examples):**

```python
>>> import dspy
>>> lm = dspy.LM("openai/gpt-4o-mini")
>>> program = dspy.Predict("question -> answer")
>>> program.set_lm(lm)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/module.py`*
