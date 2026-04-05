# /translation/complete_and_grounded.md

## dspy.evaluate.CompleteAndGrounded

```python
dspy.evaluate.CompleteAndGrounded(threshold=0.66)
```

[cite_start]**Kullanılan Yapılar (Bases):** `Module` [cite: 1]

[cite_start]Yanıtın eksiksizliğini (completeness) ve dayanaklılığını (groundedness) tek bir skorda birleştirir. [cite: 1]

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `threshold` | - | Optimizasyon sırasında kabul edilecek minimum skor. | [cite_start]`0.66` [cite: 1] |

[cite_start]*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/evaluate/auto_evaluation.py`* [cite: 1]

---

## Functions (Fonksiyonlar)

### `__call__`

```python
__call__(*args, **kwargs) -> Prediction
```

[cite_start]*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/module.py`* [cite: 1]

### `acall`

```python
acall(*args, **kwargs) -> Prediction async
```

[cite_start]*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/module.py`* [cite: 1]

### `batch`

```python
batch(examples: list[Example], num_threads: int | None = None, max_errors: int | None = None, return_failed_examples: bool = False, provide_traceback: bool | None = None, disable_progress_bar: bool = False, timeout: int = 120, straggler_limit: int = 3) -> list[Example] | tuple[list[Example], list[Example], list[Exception]]
```

[cite_start]`Parallel` modülünü kullanarak bir `dspy.Example` listesini paralel olarak işler. [cite: 1]

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `examples` | `list[Example]` | İşlenecek `dspy.Example` örneklerinin listesi. | [cite_start]**Gerekli (required)** [cite: 1] |
| `num_threads` | `int \| None` | Paralel işleme için kullanılacak iş parçacığı sayısı. | [cite_start]`None` [cite: 1] |
| `max_errors` | `int \| None` | Yürütmeyi durdurmadan önce izin verilen maksimum hata sayısı. `None` ise `dspy.settings.max_errors` değerini miras alır. | [cite_start]`None` [cite: 1] |
| `return_failed_examples` | `bool` | Başarısız örneklerin ve istisnaların döndürülüp döndürülmeyeceği. | [cite_start]`False` [cite: 1] |
| `provide_traceback` | `bool \| None` | Hata günlüklerine izleme (traceback) bilgisinin dahil edilip edilmeyeceği. | [cite_start]`None` [cite: 1] |
| `disable_progress_bar` | `bool` | İlerleme çubuğunun gösterilip gösterilmeyeceği. | [cite_start]`False` [cite: 1] |
| `timeout` | `int` | Geride kalan (straggler) bir görevin yeniden gönderilmesinden önceki saniye süresi. Devre dışı bırakmak için 0 yapın. | [cite_start]`120` [cite: 1] |
| `straggler_limit` | `int` | Yalnızca bu kadar veya daha az görev kaldığında geride kalanları kontrol eder. | [cite_start]`3` [cite: 1] |

**Dönüş Değerleri (Returns):**
* [cite_start]`list[Example] | tuple[list[Example], list[Example], list[Exception]]`: Sonuç listesi ve isteğe bağlı olarak başarısız örnekler ile istisnalar. [cite: 1]

[cite_start]*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/module.py`* [cite: 1]

### `deepcopy`

Modülün derin kopyasını (deep copy) oluşturur. [cite_start]Bu, varsayılan Python `deepcopy` metodunun özelleştirilmiş halidir; yalnızca `self.parameters()` kısmını derin kopyalar, diğer öznitelikler için yüzeysel kopya (shallow copy) yapar. [cite: 1]

[cite_start]*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/base_module.py`* [cite: 1]

### `dump_state`

```python
dump_state(json_mode=True)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/base_module.py`* [cite: 1]

### `forward`

```python
forward(example, pred, trace=None)
```

[cite_start]*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/evaluate/auto_evaluation.py`* [cite: 1]

### `get_lm`

Bu modülün tahmincileri (predictors) tarafından kullanılan dil modelini alır. Tüm tahminciler aynı LM'yi kullanıyorsa dil modelini döndürür. [cite_start]Birden fazla farklı LM kullanılıyorsa hata verir. [cite: 1]

**Dönüş Değerleri (Returns):**
* Bu modülün tahmincileri tarafından kullanılan dil modeli örneği (instance). [cite: 1]

**Hatalar (Raises):**
* [cite_start]`ValueError`: Eğer modüldeki tahminciler tarafından birden fazla farklı dil modeli kullanılıyorsa. [cite: 1]

[cite_start]*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/module.py`* [cite: 1]

### `inspect_history`

```python
inspect_history(n: int = 1, file: TextIO | None = None) -> None
```

Bu modül için LM çağrı geçmişini görüntüler. [cite_start]Hata ayıklama ve modül davranışını anlama için yararlı olan, en son yapılan dil modeli çağrılarının formatlanmış bir görünümünü yazdırır. [cite: 1]

**Parametreler:**
* `n` (`int`): Görüntülenecek son geçmiş girişlerinin sayısı. [cite_start]Varsayılan: `1`. [cite: 1]
* `file` (`TextIO | None`): Çıktının yazılacağı isteğe bağlı dosya benzeri nesne. Sağlandığında, ANSI renk kodları otomatik olarak devre dışı bırakılır. Varsayılan: `None` (standart çıktıya yazdırır). [cite: 1]

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/module.py`* [cite: 1]

### `load`

```python
load(path, allow_pickle=False, allow_unsafe_lm_state=False)
```

Kaydedilmiş modülü yükler. Sadece mevcut bir programın durumunu değil, tüm programı yüklemek istiyorsanız `dspy.load` fonksiyonuna bakabilirsiniz. [cite: 1]

**Parametreler:**
* `path` (`str`): `.json` veya `.pkl` dosyası olması gereken kaydedilmiş durum dosyasının yolu. (Gerekli) [cite_start][cite: 1]
* `allow_pickle` (`bool`): `True` ise, rastgele kod çalıştırabilen `.pkl` dosyalarının yüklenmesine izin verir. Bu tehlikelidir ve yalnızca dosyanın kaynağından eminseniz ve güvenilir bir ortamdaysanız kullanılmalıdır. [cite: 1]
* `allow_unsafe_lm_state` (`bool`): `True` ise, yüklenen durumdan güvenli olmayan LM uç nokta anahtarlarını (örn. `api_base`, `base_url` ve `model_list`) korur. [cite_start]Yalnızca güvenilir dosyalar için etkinleştirin. [cite: 1]

[cite_start]*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/base_module.py`* [cite: 1]

### `load_state`

```python
load_state(state, *, allow_unsafe_lm_state=False)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/base_module.py`* [cite: 1]

### `map_named_predictors`

Bu modüldeki tüm isimlendirilmiş tahmincilere bir fonksiyon uygular. [cite_start]Modüldeki tüm `Predict` örneklerini yineler ve her birine verilen fonksiyonu uygulayarak orijinal tahminciyi fonksiyonun dönüş değeriyle değiştirir. [cite: 1]

**Parametreler:**
* `func`: Bir `Predict` örneği alan ve yeni bir `Predict` örneği (veya uyumlu nesne) döndüren çağrılabilir bir nesne. (Gerekli) [cite_start][cite: 1]

**Dönüş Değerleri (Returns):**
* [cite_start]`Module`: Metot zincirleme (method chaining) için `self` döndürür. [cite: 1]

[cite_start]*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/module.py`* [cite: 1]

### `named_parameters`

[cite_start]PyTorch'un aksine, (yinelenmeyen) parametre listelerini de işler. [cite: 1]

[cite_start]*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/base_module.py`* [cite: 1]

### `named_predictors`

Bu modüldeki tüm isimlendirilmiş `Predict` modüllerini döndürür. Tüm parametreleri yineler ve `dspy.Predict` örneği olanları isimleriyle birlikte döndürür. [cite: 1]

**Dönüş Değerleri (Returns):**
* [cite_start]`list[tuple[str, Predict]]`: İsmin öznitelik yolu, tahmincinin ise `Predict` örneği olduğu `(isim, tahminci)` ikililerinin listesi. [cite: 1]

[cite_start]*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/module.py`* [cite: 1]

### `named_sub_modules`

```python
named_sub_modules(type_=None, skip_compiled=False) -> Generator[tuple[str, BaseModule], None, None]
```

[cite_start]Modüldeki tüm alt modülleri ve isimlerini bulur. [cite: 1]

[cite_start]*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/base_module.py`* [cite: 1]

### `parameters`

[cite_start]*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/base_module.py`* [cite: 1]

### `predictors`

Bu modüldeki tüm `Predict` modüllerini döndürür. [cite: 1]

**Dönüş Değerleri (Returns):**
* [cite_start]`list[Predict]`: Bu modüldeki tüm `Predict` örneklerinin listesi. [cite: 1]

[cite_start]*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/module.py`* [cite: 1]

### `reset_copy`

[cite_start]Modülü derin kopyalar ve tüm parametreleri sıfırlar. [cite: 1]

[cite_start]*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/base_module.py`* [cite: 1]

### `save`

```python
save(path, save_program=False, modules_to_serialize=None)
```

Modülü kaydeder. Modülü bir dizine veya dosyaya kaydedebilir. İki mod vardır: [cite: 1]
* [cite_start]`save_program=False`: Dosya uzantısına bağlı olarak modülün yalnızca durumunu (state) bir `.json` veya `.pickle` dosyasına kaydeder. [cite: 1]
* `save_program=True`: `cloudpickle` aracılığıyla hem durumu hem de mimariyi içeren tüm modülü bir dizine kaydeder. [cite: 1]

**Parametreler:**
* [cite_start]`path` (`str`): Kayıt dosyası yolu. [cite: 1]
* [cite_start]`save_program` (`bool`): `True` ise tüm modülü kaydeder, aksi takdirde sadece durumu kaydeder. [cite: 1]
* [cite_start]`modules_to_serialize` (`list`): `cloudpickle` ile serileştirilecek modüllerin listesi. [cite: 1]

[cite_start]*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/base_module.py`* [cite: 1]

### `set_lm`

Bu modüldeki tüm tahminciler için dil modelini ayarlar. Bu metot, modül içinde bulunan tüm `Predict` örnekleri için dil modelini yinelemeli (recursively) olarak ayarlar. [cite: 1]

**Parametreler:**
* `lm`: Tüm tahminciler için kullanılacak dil modeli örneği. (Gerekli) [cite_start][cite: 1]

[cite_start]*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/module.py`* [cite: 1]

