# /translation/python_interpreter.md

## dspy.PythonInterpreter

```python
dspy.PythonInterpreter(deno_command: list[str] | None = None, enable_read_paths: list[PathLike | str] | None = None, enable_write_paths: list[PathLike | str] | None = None, enable_env_vars: list[str] | None = None, enable_network_access: list[str] | None = None, sync_files: bool = True, tools: dict[str, Callable[..., str]] | None = None, output_fields: list[dict] | None = None)
```

Deno ve Pyodide kullanarak güvenli Python yürütmesi (execution) için yerel yorumlayıcı (interpreter).



WASM tabanlı bir korumalı alanda (sandbox) güvenli kod yürütmesi için `Interpreter` protokolünü uygular. Kod, varsayılan olarak ana makine (host) dosya sistemine, ağına veya ortamına erişimi olmayan yalıtılmış bir Pyodide ortamında çalışır.

### Önkoşullar (Prerequisites)
Deno kurulu olmalıdır: https://docs.deno.com/runtime/getting_started/installation/

**Örnekler (Examples):**

```python
# Temel yürütme (Basic execution)
with PythonInterpreter() as interp:
    result = interp("print(1 + 2)")  # "3" döndürür

# Ana makine tarafı (host-side) araçlarıyla
def my_tool(question: str) -> str:
    return "answer"

with PythonInterpreter(tools={"my_tool": my_tool}) as interp:
    result = interp("print(my_tool(question='test'))")
```

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `deno_command` | `list[str] \| None` | Deno'yu başlatmak için komut listesi. | `None` |
| `enable_read_paths` | `list[PathLike \| str] \| None` | Korumalı alanda (sandbox) okumaya izin verilecek dosyalar veya dizinler. | `None` |
| `enable_write_paths` | `list[PathLike \| str] \| None` | Korumalı alanda yazmaya izin verilecek dosyalar veya dizinler. Tüm yazma yolları, bağlama (mounting) işlemi için aynı zamanda okunabilir olacaktır. | `None` |
| `enable_env_vars` | `list[str] \| None` | Korumalı alanda izin verilecek ortam değişkeni (environment variable) adları. | `None` |
| `enable_network_access` | `list[str] \| None` | Korumalı alanda ağ erişimine izin verilecek alan adları (domains) veya IP'ler. | `None` |
| `sync_files` | `bool` | Ayarlanırsa, yürütmeden sonra korumalı alan içindeki değişiklikleri orijinal dosyalarla senkronize eder. | `True` |
| `tools` | `dict[str, Callable[..., str]] \| None` | Araç adlarını çağrılabilir fonksiyonlara eşleyen sözlük. Her fonksiyon anahtar kelime argümanlarını kabul etmeli ve bir string döndürmelidir. Araçlar, doğrudan korumalı alan kodundan adlarıyla çağrılabilir. | `None` |
| `output_fields` | `list[dict] \| None` | Yazılı (typed) SUBMIT imzası için çıktı alanı tanımlarının listesi. Her sözlükte 'name' (isim) ve isteğe bağlı olarak 'type' (tür) anahtarları olmalıdır. | `None` |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/python_interpreter.py`*

---

## Functions (Fonksiyonlar)

### `__call__`

```python
__call__(code: str, variables: dict[str, Any] | None = None) -> Any
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/python_interpreter.py`*

### `execute`

```python
execute(code: str, variables: dict[str, Any] | None = None) -> Any
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/python_interpreter.py`*

### `shutdown`

```python
shutdown() -> None
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/python_interpreter.py`*

### `start`

```python
start() -> None
```

Deno/Pyodide korumalı alanını (sandbox) başlatır.

Bu, Deno alt sürecini (subprocess) başlatarak korumalı alanı önceden ısıtır (pre-warms). Havuza alma (pooling) için açıkça çağrılabilir veya ilk `execute()` işleminde tembel (lazy) olarak çağrılır.

**Eşkuvvetli (Idempotent):** birden çok kez çağrılması güvenlidir.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/primitives/python_interpreter.py`*
