# /translation/configure.md

## dspy.configure

DSPy için varsayılan dil modelini, adaptörü ve diğer ayarları yapılandırır.

```python
dspy.configure(**kwargs)
```

Betiğinizin (script) veya not defterinizin (notebook) en üstüne yakın bir yerde `dspy.configure(...)` fonksiyonunu bir kez çağırın. `dspy.context` ile geçersiz kılmadığınız (override) sürece her DSPy modülü bu varsayılanları kullanacaktır. Değerler, siz tekrar `dspy.configure(...)` çağırana kadar kalıcı olur.

> **Not:** Çıplak bir model string'i (metni) yerine `lm` parametresine bir `dspy.LM` nesnesi (object) aktarın.

### Ayarlar (Settings)

| Ayar (Setting) | Varsayılan (Default) | Açıklama (Description) |
| :--- | :--- | :--- |
| `lm` | `None` | Varsayılan dil modeli. Bir `dspy.LM` örneği aktarın. |
| `adapter` | `None` | İstemleri formatlar (biçimlendirir) ve LM yanıtlarını ayrıştırır (parses). `None` olduğunda, modüller `dspy.ChatAdapter` kullanır. |
| `callbacks` | `[]` | Gözlemlenebilirlik (Observability) ve günlükleme (logging) kancaları (hooks). Gözlemlenebilirlik bölümüne bakın. |
| `track_usage` | `False` | Her LM çağrısı için token sayılarını kaydeder. |
| `async_max_workers` | `8` | Asenkron işlemler için maksimum eşzamanlı (concurrent) çalışan (worker) sayısı. |
| `num_threads` | `8` | `dspy.Parallel` için iş parçacığı (thread) sayısı. |
| `max_errors` | `10` | Bu kadar hatadan sonra paralel yürütmeyi durdurur. |
| `disable_history` | `False` | LM çağrı geçmişini kaydetmeyi durdurur. |
| `max_history_size` | `10000` | Depolanan geçmiş girdileri (entries) için üst sınır. |
| `allow_tool_async_sync_conversion` | `False` | Asenkron araçların senkron kodda çalışmasına izin verir. Asenkron (Async) bölümüne bakın. |
| `provide_traceback` | `False` | Hata günlüklerine Python izlemelerini (tracebacks) dahil eder. |
| `warn_on_type_mismatch` | `True` | Bir modül girdi türü, imzayla (signature) eşleşmediğinde uyarır. |

---

## Örnekler (Examples)

**Varsayılan LM'i ayarlama:**
```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

qa = dspy.Predict("question -> answer")
result = qa(question="What is the capital of France?")
print(result.answer)
```

**LM'i ve adaptörü ayarlama:**
```python
import dspy

dspy.configure(
    lm=dspy.LM("anthropic/claude-3-5-sonnet-20241022"),
    adapter=dspy.JSONAdapter(),
)
```

**Kullanım izlemeyi (usage tracking) etkinleştirme ve eşzamanlılığı (concurrency) ayarlama:**
```python
import dspy

dspy.configure(
    lm=dspy.LM("gemini/gemini-2.5-flash"),
    track_usage=True,
    async_max_workers=4,
)
```

---

## `dspy.configure` Ne Zaman Kullanılmalı?

Programınızın çoğuna (betikler, not defterleri, test kurulumu veya uygulama başlangıcı) bir varsayılan ayar grubunun uygulanması gerektiğinde `dspy.configure(...)` kullanın.

Tek bir çağrı veya tek bir blok için farklı ayarlara ihtiyacınız varsa, bunun yerine `dspy.context` kullanın.

## İş Parçacığı Güvenliği (Thread Safety)

Yalnızca `dspy.configure(...)` fonksiyonunu ilk çağıran iş parçacığı onu tekrar çağırabilir. Bunu deneyen diğer iş parçacıkları bir `RuntimeError` alacaktır. Asenkron kodda, yalnızca `dspy.configure(...)` fonksiyonunu ilk çağıran görev (task) onu çağırmaya devam edebilir.

Çalışan iş parçacıkları (worker threads), asenkron görevler veya `dspy.Parallel` blokları içindeki geçici geçersiz kılmalar (temporary overrides) için `dspy.context` kullanın.

---

## Ayrıca Bakınız (See Also)
* **`dspy.context`** — tek bir blok boyunca süren geçici geçersiz kılmalar.  https://dspy.ai/api/utils/context/
* **`dspy.LM`** — `lm` olarak aktaracağınız dil modelini oluşturur.  https://dspy.ai/api/models/LM/
* **Dil Modelleri (Language Models)** — LM yapılandırmasına genel bakış.  https://dspy.ai/learn/programming/language_models/
* **Adaptörler (Adapters)** — adaptörlerin istemleri nasıl formatladığı ve yanıtları nasıl ayrıştırdığı.  https://dspy.ai/learn/programming/adapters/
