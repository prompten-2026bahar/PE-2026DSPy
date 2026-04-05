# /translation/configure_cache.md

## dspy.configure_cache

```python
dspy.configure_cache(enable_disk_cache: bool | None = True, enable_memory_cache: bool | None = True, disk_cache_dir: str | None = DISK_CACHE_DIR, disk_size_limit_bytes: int | None = DISK_CACHE_LIMIT, memory_max_entries: int = 1000000)
```

DSPy için önbelleği (cache) yapılandırır.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `enable_disk_cache` | `bool \| None` | Disk üzeri (on-disk) önbelleğin etkinleştirilip etkinleştirilmeyeceği. | `True` |
| `enable_memory_cache` | `bool \| None` | Bellek içi (in-memory) önbelleğin etkinleştirilip etkinleştirilmeyeceği. | `True` |
| `disk_cache_dir` | `str \| None` | Disk üzeri önbelleğin saklanacağı dizin. | `DISK_CACHE_DIR` |
| `disk_size_limit_bytes` | `int \| None` | Disk üzeri önbelleğin boyut sınırı. | `DISK_CACHE_LIMIT` |
| `memory_max_entries` | `int` | Bellek içi önbellekteki maksimum girdi (entry) sayısı. Önbelleğin sınırsız büyümesine izin vermek için bu parametreyi `math.inf` veya benzer bir değere ayarlayın. | `1000000` |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/clients/__init__.py`*
