# /translation/asyncify.md

## dspy.asyncify

```python
dspy.asyncify(program: Module) -> Callable[[Any, Any], Awaitable[Any]]
```

Bir DSPy programını asenkron olarak çağrılabilecek şekilde sarmalar (wraps). Bu, bir programı başka bir görevle (örneğin, başka bir DSPy programıyla) paralel olarak çalıştırmak için oldukça kullanışlıdır.

Bu uygulama (implementation), geçerli iş parçacığının (thread) yapılandırma bağlamını (configuration context) çalışan iş parçacığına (worker thread) aktarır (propagates).

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `program` | `Module` | Asenkron yürütme (execution) için sarmalanacak DSPy programı. | **Gerekli (required)** |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `Callable[[Any, Any], Awaitable[Any]]` | **Asenkron bir fonksiyon:** Bekletildiğinde (`await` edildiğinde), programı bir çalışan iş parçacığında (worker thread) çalıştıran asenkron bir fonksiyon. Geçerli iş parçacığının yapılandırma bağlamı her çağrı için devralınır (inherited). |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/utils/asyncify.py`*
