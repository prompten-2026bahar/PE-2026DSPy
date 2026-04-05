# /translation/code_act.md

## dspy.CodeAct

```python
dspy.CodeAct(signature: str | type[Signature], tools: list[Callable], max_iters: int = 5, interpreter: PythonInterpreter | None = None)
```

**Kullanılan Yapılar (Bases):** `ReAct`, `ProgramOfThought`

CodeAct, sorunu çözmek için Kod Yorumlayıcıyı (Code Interpreter) ve önceden tanımlanmış araçları kullanan bir modüldür.
CodeAct sınıfını belirtilen model, sıcaklık ve maksimum token değerleriyle başlatır.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `signature` | `Union[str, Type[Signature]]` | Modülün imzası. | **Gerekli (required)** |
| `tools` | `list[Callable]` | Kullanılacak çağrılabilir araçlar (tool callables). CodeAct çağrılabilir nesneleri (callable objects) değil, yalnızca fonksiyonları kabul eder. | **Gerekli (required)** |
| `max_iters` | `int` | Yanıtı üretmek için maksimum yineleme (iteration) sayısı. | `5` |
| `interpreter` | `PythonInterpreter \| None` | Kullanılacak PythonInterpreter örneği. `None` ise yeni bir tane oluşturulur. | `None` |

**Örnekler (Examples):**

```python
from dspy.predict import CodeAct

def factorial(n):
    if n == 1:
        return 1
    return n * factorial(n-1)

act = CodeAct("n->factorial", tools=[factorial])
act(n=5) # 120
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/predict/code_act.py`*

---

## Functions (Fonksiyonlar)

### `__call__`
```python
__call__(*args, **kwargs) -> Prediction
```
*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/module.py`*

### `batch`
```python
batch(examples: list[Example], num_threads: int | None = None, max_errors: int | None = None, return_failed_examples: bool = False, provide_traceback: bool | None = None, disable_progress_bar: bool = False, timeout: int = 120, straggler_limit: int = 3) -> list[Example] | tuple[list[Example], list[Example], list[Exception]]
```
`Parallel` modülünü kullanarak bir `dspy.Example` örnekleri listesini paralel olarak işler.

**Parametreler:**
* `examples`: İşlenecek `dspy.Example` örneklerinin listesi. (Gerekli)
* Diğer yapılandırma ayarları (num_threads, max_errors vb.) `dspy.settings` üzerinden miras alınır veya belirtilen değerlerle ezilir.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/module.py`*

### `deepcopy`
Modülün derin kopyasını oluşturur. Yalnızca `self.parameters()` kısmını derin kopyalar.
*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/base_module.py`*

### `dump_state`
```python
dump_state(json_mode=True)
```
*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/base_module.py`*

### `get_lm`
Bu modülün tahmincileri tarafından kullanılan dil modelini döndürür.
*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/module.py`*

### `inspect_history`
```python
inspect_history(n: int = 1, file: TextIO | None = None) -> None
```
Bu modül için LM çağrı geçmişini görüntüler. Hata ayıklama için kullanışlıdır.
*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/module.py`*

### `load` / `load_state`
Modülün durumunu veya tüm mimarisini yüklemek için kullanılır.
*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/base_module.py`*

### `map_named_predictors`
Modüldeki tüm isimlendirilmiş tahmincilere bir fonksiyon uygular.
*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/module.py`*

### `named_parameters` / `named_predictors` / `named_sub_modules`
Modül içerisindeki parametreleri, tahmincileri ve alt modülleri listeler.
*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/base_module.py`*

### `parameters` / `predictors`
Modüldeki yapıları döndürür.
*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/base_module.py`*

### `reset_copy`
Modülü derin kopyalar ve parametreleri sıfırlar.
*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/base_module.py`*

### `save`
```python
save(path, save_program=False, modules_to_serialize=None)
```
Modülü kaydeder. İki modu vardır: `save_program=False` (yalnızca durumu kaydeder) ve `save_program=True` (tüm modülü dizine kaydeder).
*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/base_module.py`*

### `set_lm`
```python
set_lm(lm)
```
Tahminciler için kullanılacak dil modelini ayarlar.
*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/module.py`*

---

## CodeAct

CodeAct, sorunları çözmek için kod üretimini (code generation) araç yürütme (tool execution) ile birleştiren bir DSPy modülüdür. Görevleri yerine getirmek için sağlanan araçları ve Python standart kütüphanesini kullanan Python kod parçacıkları üretir.

### Basic Usage (Temel Kullanım)

İşte CodeAct'i kullanmanın basit bir örneği:

```python
import dspy
from dspy.predict import CodeAct

# Basit bir araç fonksiyonu tanımlayın
def factorial(n: int) -> int:
    '''Bir sayının faktöriyelini hesaplar.'''
    if n == 1:
        return 1
    return n * factorial(n-1)

# Bir CodeAct örneği oluşturun
act = CodeAct("n->factorial_result", tools=[factorial])

# CodeAct örneğini kullanın
result = act(n=5)
print(result) # factorial(5) = 120 hesaplayacaktır
```

### How It Works (Nasıl Çalışır)

CodeAct yinelemeli (iterative) bir şekilde çalışır:

1. Girdi parametrelerini ve mevcut araçları alır.
2. Bu araçları kullanan Python kod parçacıkları üretir.
3. Bir Python korumalı alanı (sandbox) kullanarak kodu çalıştırır.
4. Çıktıyı toplar ve görevin tamamlanıp tamamlanmadığını belirler.
5. Toplanan bilgilere dayanarak orijinal soruyu yanıtlar.

### ⚠️ Limitations (Sınırlamalar)

#### Yalnızca saf fonksiyonları araç olarak kabul eder (çağrılabilir nesneler kabul edilmez)
Çağrılabilir bir nesnenin (callable object) kullanılması nedeniyle aşağıdaki örnek çalışmaz.

```python
# ❌ NG (Çalışmaz)
class Add():
    def __call__(self, a: int, b: int):
        return a + b

dspy.CodeAct("question -> answer", tools=[Add()])
```

#### Harici kütüphaneler kullanılamaz
Harici `numpy` kütüphanesinin kullanılması nedeniyle aşağıdaki örnek çalışmaz.

```python
# ❌ NG (Çalışmaz)
import numpy as np

def exp(i: int):
    return np.exp(i)

dspy.CodeAct("question -> answer", tools=[exp])
```

#### Tüm bağımlı fonksiyonların `CodeAct`'e iletilmesi gerekir
CodeAct'e iletilmeyen diğer fonksiyonlara veya sınıflara bağlı olan fonksiyonlar kullanılamaz. Aşağıdaki örnek çalışmaz çünkü araç fonksiyonları `Profile` veya `secret_function` gibi CodeAct'e iletilmeyen diğer sınıf/fonksiyonlara bağımlıdır.

```python
# ❌ NG (Çalışmaz)
from pydantic import BaseModel

class Profile(BaseModel):
    name: str
    age: int

def age(profile: Profile):
    return

def parent_function():
    print("Hi!")

def child_function():
    parent_function()

dspy.CodeAct("question -> answer", tools=[age, child_function])
```

Bunun yerine, gerekli tüm araç fonksiyonları `CodeAct`'e iletildiği için aşağıdaki örnek sorunsuz çalışır:

```python
# ✅ OK (Çalışır)
def parent_function():
    print("Hi!")

def child_function():
    parent_function()

dspy.CodeAct("question -> answer", tools=[parent_function, child_function])
```
