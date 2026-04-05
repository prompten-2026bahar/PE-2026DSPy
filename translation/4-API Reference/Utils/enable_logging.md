# /translation/enable_logging.md

## dspy.enable_logging

```python
dspy.enable_logging()
```

DSPy genelindeki olay günlükleme (event logging) API'leri (`eprint()`, `logger.info()` vb.) tarafından kullanılan `DSPyLoggingStream`'i etkinleştirerek sonraki tüm olay günlüklerini yayınlar (emit). Bu, `disable_logging()` işleminin etkilerini tersine çevirir.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/utils/logging_utils.py`*
