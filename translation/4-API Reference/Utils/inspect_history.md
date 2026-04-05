# /translation/inspect_history.md

## dspy.inspect_history

```python
dspy.inspect_history(n: int = 1, file: TextIO | None = None) -> None
```

Tüm LM'ler (Dil Modelleri) arasında paylaşılan küresel (global) geçmiş.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `n` | `int` | Görüntülenecek son girdi (entry) sayısı. Varsayılan 1'dir. | `1`
| `file` | `TextIO \| None` | Çıktıyı yazmak için isteğe bağlı dosya benzeri (file-like) bir nesne. Sağlandığında, ANSI renk kodları otomatik olarak devre dışı bırakılır. Varsayılan `None`'dır (standart çıktıya/stdout yazdırır). | `None`

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/clients/base_lm.py`*
