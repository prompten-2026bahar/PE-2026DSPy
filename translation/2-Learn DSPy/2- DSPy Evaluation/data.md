# Veri (Data)

DSPy bir makine öğrenmesi çerçevesidir, bu nedenle onunla çalışmak; eğitim setleri (training sets), geliştirme setleri (development sets) ve test setlerini içerir. Verilerinizdeki her bir örnek için tipik olarak üç tür değer arasında ayrım yaparız: girdiler, ara etiketler ve nihai etiket. DSPy'yi herhangi bir ara veya nihai etiket olmadan etkili bir şekilde kullanabilirsiniz, ancak en azından birkaç örnek girdiye ihtiyacınız olacaktır.


## DSPy `Example` Nesneleri

DSPy'deki verilerin temel veri türü `Example`dır. Eğitim setinizdeki ve test setinizdeki öğeleri temsil etmek için **Example** (Örnek) nesnelerini kullanacaksınız.

DSPy **Example** nesneleri, Python'daki `dict` (sözlük) yapılarına benzer ancak birkaç yararlı yardımcı araca sahiptir. DSPy modülleriniz, `Example`ın özel bir alt sınıfı olan `Prediction` türünde değerler döndürecektir.

DSPy kullandığınızda, birçok değerlendirme ve optimizasyon çalışması yapacaksınız. Bireysel veri noktalarınız `Example` türünde olacaktır:

```python
qa_pair = dspy.Example(question="This is a question?", answer="This is an answer.")

print(qa_pair)
print(qa_pair.question)
print(qa_pair.answer)
```
**Çıktı:**
```text
Example({'question': 'This is a question?', 'answer': 'This is an answer.'}) (input_keys=None)
This is a question?
This is an answer.
```

Örnekler herhangi bir alan anahtarına (key) ve herhangi bir değer türüne sahip olabilir, ancak genellikle değerler dizgi (string) biçimindedir.

```text
object = Example(field1=value1, field2=value2, field3=value3, ...)
```

Eğitim setinizi şu şekilde ifade edebilirsiniz:

```python
trainset = [dspy.Example(report="LONG REPORT 1", summary="short summary 1"), ...]
```


### Girdi Anahtarlarını Belirleme
Geleneksel makine öğrenmesinde "girdiler" ve "etiketler" birbirinden ayrılmıştır.

DSPy'de `Example` nesneleri, belirli alanları girdi olarak işaretleyebilen `with_inputs()` metoduna sahiptir. (Geriye kalanlar sadece meta veri veya etiketlerdir.)

```python
# Tek Girdi.
print(qa_pair.with_inputs("question"))

# Çoklu Girdi; etiketlerinizi gerçekten istemedikçe girdi olarak işaretlemeye dikkat edin.
print(qa_pair.with_inputs("question", "answer"))
```

Değerlere `.` (nokta) operatörü kullanılarak erişilebilir. `Example(name="John Doe", job="sleep")` şeklinde tanımlanmış bir nesnedeki `name` anahtarının değerine `nesne.name` üzerinden erişebilirsiniz.

Belirli anahtarlara erişmek veya onları hariç tutmak için, sırasıyla yalnızca girdi veya girdi olmayan anahtarları içeren yeni Example nesneleri döndüren `inputs()` ve `labels()` metotlarını kullanın.

```python
article_summary = dspy.Example(article= "This is an article.", summary= "This is a summary.").with_inputs("article")

input_key_only = article_summary.inputs()
non_input_key_only = article_summary.labels()

print("Example object with Input fields only:", input_key_only)
print("Example object with Non-Input fields only:", non_input_key_only)
```

**Çıktı**
```
Example object with Input fields only: Example({'article': 'This is an article.'}) (input_keys={'article'})
Example object with Non-Input fields only: Example({'summary': 'This is a summary.'}) (input_keys=None)
```

<!-- ## Kaynaklardan Veri Seti Yükleme

DSPy'de veri setlerini içe aktarmanın en kolay yollarından biri `DataLoader` kullanmaktır. İlk adım bir nesne tanımlamaktır; bu nesne daha sonra farklı formatlardaki veri setlerini yüklemek için yardımcı araçları çağırmak amacıyla kullanılabilir:

```python
from dspy.datasets import DataLoader

dl = DataLoader()
```

Çoğu veri seti formatı için işlem oldukça basittir; dosya yolunu ilgili formatın metoduna iletirsiniz ve karşılığında veri seti için bir Example listesi alırsınız:

```python
import pandas as pd

csv_dataset = dl.from_csv(
    "sample_dataset.csv",
    fields=("instruction", "context", "response"),
    input_keys=("instruction", "context")
)

json_dataset = dl.from_json(
    "sample_dataset.json",
    fields=("instruction", "context", "response"),
    input_keys=("instruction", "context")
)

parquet_dataset = dl.from_parquet(
    "sample_dataset.parquet",
    fields=("instruction", "context", "response"),
    input_keys=("instruction", "context")
)

pandas_dataset = dl.from_pandas(
    pd.read_csv("sample_dataset.csv"),    # DataFrame
    fields=("instruction", "context", "response"),
    input_keys=("instruction", "context")
)
```

Bunlar, DataLoader'ın doğrudan dosyadan yüklemeyi desteklediği bazı formatlardır. Arka planda, bu metodların çoğu bu formatları yüklemek için datasets kütüphanesindeki load_dataset metodunu kullanır. Ancak metin verileriyle çalışırken genellikle HuggingFace veri setlerini kullanırsınız; HF veri setlerini Example listesi formatında içe aktarmak için from_huggingface metodunu kullanabiliriz:

```python
blog_alpaca = dl.from_huggingface(
    "intertwine-expel/expel-blog",
    input_keys=("title",)
)
```

İlgili anahtara erişerek veri setinin bölümlerine (splits) ulaşabilirsiniz:

```python
train_split = blog_alpaca['train']

# Since this is the only split in the dataset we can split this into 
# train and test split ourselves by slicing or sampling 75 rows from the train
# split for testing.
testset = train_split[:75]
trainset = train_split[75:]
```

load_dataset kullanarak bir HuggingFace veri setini nasıl yüklüyorsanız, from_huggingface aracılığıyla da tam olarak o şekilde yüklersiniz. Buna belirli bölümleri, alt bölümleri, okuma talimatlarını vb. iletmek dahildir. Kod parçacıkları için HF'den yükleme ile ilgili kopya kağıdı parçacıklarına başvurabilirsiniz. -->