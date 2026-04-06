Bu eğitimde, DSPy kullanan [PAPILLON yazarlarına ait bu eğitime](https://colab.research.google.com/github/Columbia-NLP-Lab/PAPILLON/blob/main/papillon_tutorial.ipynb) başvurun.

Bu eğitim, DSPy kullanımının daha ileri bir bağlamdaki birkaç yönünü göstermektedir:

1. Harici bir araç kullanan küçük bir yerel LM içeren çok aşamalı bir `dspy.Module` oluşturur.
2. DSPy içinde çok aşamalı bir _judge_ oluşturur ve bunu değerlendirme için bir metrik olarak kullanır.
3. Küçük bir yerel LM için öğretmen olarak büyük bir model kullanarak, bu _judge_ ile `dspy.Module`'ü optimize eder.
