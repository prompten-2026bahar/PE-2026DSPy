# /translation/program_of_thought.md

## dspy.ProgramOfThought

```python
dspy.ProgramOfThought(signature: str | type[Signature], max_iters: int = 3, interpreter: PythonInterpreter | None = None)
```

**Kullanılan Yapılar (Bases):** `Module`

Bir problemi çözmek için Python programları çalıştıran bir DSPy modülüdür. Bu modül `deno` kurulumunu gerektirir. Lütfen https://docs.deno.com/runtime/getting_started/installation/ adresini takip ederek `deno`'yu kurun.

**Örnekler (Examples):**

```python
import dspy

lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)

pot = dspy.ProgramOfThought("question -> answer")
pot(question="what is 1+1?")
```

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `signature` | `str \| type[Signature]` | Modülün imzası. | **Gerekli (required)** |
| `max_iters` | `int` | Kod üretimini (code generation) ve yürütmesini (execution) yeniden denemek için maksimum yineleme sayısı. | `3` |
| `interpreter` | `PythonInterpreter \| None` | Kullanılacak PythonInterpreter örneği. `None` ise yeni bir tane oluşturulur. | `None` |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/predict/program_of_thought.py`*

---

## Functions (Fonksiyonlar)

### `__call__`

```python
__call__(*args, **kwargs) -> Prediction
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/module.py`*

### `acall`

```python
acall(*args, **kwargs) -> Prediction async
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/module.py`*

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

### `forward`

```python
forward(**kwargs)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/predict/program_of_thought.py`*

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

### `inspect_history`

```python
inspect_history(n: int = 1, file: TextIO | None = None) -> None
```

Bu modül için LM çağrı geçmişini görüntüler.
Hata ayıklama ve modül davranışını anlama için yararlı olan, en son yapılan dil modeli çağrılarının formatlanmış bir görünümünü yazdırır.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `n` | `int` | Görüntülenecek son geçmiş girişlerinin sayısı. Varsayılan 1. | `1` |
| `file` | `TextIO \| None` | Çıktının yazılacağı isteğe bağlı dosya benzeri nesne. Sağlandığında, ANSI renk kodları otomatik olarak devre dışı bırakılır. Varsayılan `None` (standart çıktıya yazdırır). | `None` |

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

### `map_named_predictors`

Bu modüldeki tüm isimlendirilmiş tahmincilere bir fonksiyon uygular.
Bu metot, modüldeki tüm `Predict` örneklerini yineler ve her birine verilen fonksiyonu uygulayarak orijinal tahminciyi fonksiyonun dönüş değeriyle değiştirir.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `func` | - | Bir `Predict` örneği alan ve yeni bir `Predict` örneği (veya uyumlu nesne) döndüren çağrılabilir bir nesne. | **Gerekli (required)** |

**Dönüş Değerleri (Returns):**

| İsim (Name) | Tip (Type) | Açıklama (Description) |
| :--- | :--- | :--- |
| - | `Module` | Metot zincirleme (method chaining) için `self` döndürür. |

**Örnekler (Examples):**

```python
>>> import dspy
>>> class MyProgram(dspy.Module):
...     def __init__(self):
...         super().__init__()
...         self.qa = dspy.Predict("question -> answer")
...
>>> program = MyProgram()
>>> program.map_named_predictors(lambda p: p)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/module.py`*

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

### `named_sub_modules`

```python
named_sub_modules(type_=None, skip_compiled=False) -> Generator[tuple[str, BaseModule], None, None]
```

Modüldeki tüm alt modülleri ve isimlerini bulur.
Diyelim ki `self.children[4]['key'].sub_module` bir alt modüldür. O zaman isim `children[4]['key'].sub_module` olacaktır. Ancak alt modül farklı yollardan erişilebilir durumdaysa, yollardan yalnızca biri döndürülür.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/base_module.py`*

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
