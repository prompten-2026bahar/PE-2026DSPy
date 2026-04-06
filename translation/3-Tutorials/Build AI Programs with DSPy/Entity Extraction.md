# Eğitim: Varlık Çıkarma

Bu eğitim, DSPy ile CoNLL-2003 veri kümesini kullanarak **varlık çıkarma** işleminin nasıl yapılacağını gösterir. Odak noktamız, kişilere atıfta bulunan varlıkların çıkarılmasıdır. Şunları yapacağız:

- CoNLL-2003 veri kümesinden kişilere atıfta bulunan varlıkları çıkarıp etiketleyeceğiz
- Kişilere atıfta bulunan varlıkları çıkarmak için bir DSPy programı tanımlayacağız
- Programı CoNLL-2003 veri kümesinin bir alt kümesi üzerinde optimize edip değerlendireceğiz

Bu eğitimin sonunda, görevleri DSPy içinde imzalar ve modüller kullanarak nasıl yapılandıracağınızı, sisteminizin performansını nasıl değerlendireceğinizi ve optimizasyon araçlarıyla kalitesini nasıl artıracağınızı anlayacaksınız.

DSPy’nin en güncel sürümünü kurup birlikte ilerleyin. Eğer DSPy’nin kavramsal bir genel bakışını arıyorsanız, bunun yerine bu [yakın tarihli ders](https://www.youtube.com/live/JEMYuzrKLUw) başlamak için iyi bir yerdir.

```python
# DSPy'nin en güncel sürümünü kurun
%pip install -U dspy
# CoNLL-2003 veri kümesini yüklemek için Hugging Face datasets kütüphanesini kurun
%pip install datasets
```

<details>
<summary>Önerilir: Arka planda neler olduğunu anlamak için MLflow Tracing kurun.</summary>

### MLflow DSPy Entegrasyonu

<a href="https://mlflow.org/">MLflow</a>, DSPy ile doğal olarak entegre olan ve açıklanabilirlik ile deney takibi sunan bir LLMOps aracıdır. Bu eğitimde, istemleri ve optimizasyon ilerlemesini izler olarak görselleştirmek için MLflow kullanabilir, böylece DSPy’nin davranışını daha iyi anlayabilirsiniz. Aşağıdaki dört adımı izleyerek MLflow’u kolayca kurabilirsiniz.

![MLflow Trace](./mlflow-tracing-entity-extraction.png)

1. MLflow’u kurun

```bash
%pip install mlflow>=2.20
```

2. Ayrı bir terminalde MLflow UI’ı başlatın
```bash
mlflow ui --port 5000
```

3. Notebook’u MLflow’a bağlayın
```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("DSPy")
```

4. İzlemeyi etkinleştirin.
```python
mlflow.dspy.autolog()
```

Entegrasyon hakkında daha fazla bilgi edinmek için [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) sayfasını da ziyaret edin.
</details>

## Veri Kümesini Yükleme ve Hazırlama

Bu bölümde, varlık çıkarma görevlerinde yaygın olarak kullanılan CoNLL-2003 veri kümesini hazırlıyoruz. Veri kümesi; kişiler, kurumlar ve konumlar gibi varlık etiketleriyle açıklanmış token’lar içerir.

Şunları yapacağız:
1. Veri kümesini Hugging Face `datasets` kütüphanesini kullanarak yükleyeceğiz.
2. Kişilere atıfta bulunan token’ları çıkarmak için bir fonksiyon tanımlayacağız.
3. Eğitim ve test için daha küçük alt kümeler oluşturmak üzere veri kümesini dilimleyeceğiz.

DSPy, örneklerin yapılandırılmış bir biçimde olmasını bekler; bu yüzden kolay entegrasyon için veri kümesini DSPy `Examples` biçimine de dönüştüreceğiz.

```python
import os
import tempfile
from datasets import load_dataset
from typing import Dict, Any, List
import dspy

def load_conll_dataset() -> dict:
    """
    CoNLL-2003 veri kümesini train, validation ve test bölümleriyle yükler.
    
    Returns:
        dict: 'train', 'validation' ve 'test' anahtarlarına sahip veri kümesi bölümleri.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Varsayılan Hugging Face önbellek dizinini desteklemeyen bazı barındırılan notebook
        # ortamlarıyla uyumluluk için geçici bir Hugging Face önbellek dizini kullan
        os.environ["HF_DATASETS_CACHE"] = temp_dir
        return load_dataset("conll2003", trust_remote_code=True)

def extract_people_entities(data_row: dict[str, Any]) -> list[str]:
    """
    CoNLL-2003 veri kümesindeki bir satırdan kişilere atıfta bulunan varlıkları çıkarır.
    
    Args:
        data_row (dict[str, Any]): Token'ları ve NER etiketlerini içeren bir veri kümesi satırı.
    
    Returns:
        list[str]: Kişi olarak etiketlenmiş token'ların listesi.
    """
    return [
        token
        for token, ner_tag in zip(data_row["tokens"], data_row["ner_tags"])
        if ner_tag in (1, 2)  # CoNLL varlık kodları 1 ve 2 kişilere karşılık gelir
    ]

def prepare_dataset(data_split, start: int, end: int) -> list[dspy.Example]:
    """
    DSPy ile kullanılmak üzere dilimlenmiş bir veri kümesi bölümünü hazırlar.
    
    Args:
        data_split: Veri kümesi bölümü (ör. train veya test).
        start (int): Dilimin başlangıç indeksi.
        end (int): Dilimin bitiş indeksi.
    
    Returns:
        list[dspy.Example]: Token'ları ve beklenen etiketleri içeren DSPy Example listesi.
    """
    return [
        dspy.Example(
            tokens=row["tokens"],
            expected_extracted_people=extract_people_entities(row)
        ).with_inputs("tokens")
        for row in data_split.select(range(start, end))
    ]

# Veri kümesini yükle
dataset = load_conll_dataset()

# Eğitim ve test kümelerini hazırla
train_set = prepare_dataset(dataset["train"], 0, 50)
test_set = prepare_dataset(dataset["test"], 0, 200)
```

## DSPy’yi Yapılandırın ve Bir Varlık Çıkarma Programı Oluşturun

Burada, token’laştırılmış metinden kişilere atıfta bulunan varlıkları çıkarmak için bir DSPy programı tanımlıyoruz.

Ardından, DSPy’yi programın tüm çağrılarında belirli bir dil modelini (`gpt-4o-mini`) kullanacak şekilde yapılandırıyoruz.

**Tanıtılan Temel DSPy Kavramları:**
- **İmzalar (Signatures):** Programınız için yapılandırılmış girdi/çıktı şemalarını tanımlar.
- **Modüller (Modules):** Program mantığını yeniden kullanılabilir, birleştirilebilir birimler içinde kapsüller.

Özellikle şunları yapacağız:
- Girdi (`tokens`) ve çıktı (`extracted_people`) alanlarını belirtmek için bir `PeopleExtraction` DSPy Signature oluşturacağız.
- `PeopleExtraction` imzasını uygulamak için DSPy’nin yerleşik `dspy.ChainOfThought` modülünü kullanan bir `people_extractor` programı tanımlayacağız. Program, dil modeli (LM) istemleri kullanarak girdi token listesi içinden kişilere atıfta bulunan varlıkları çıkarır.
- Programı çağırırken DSPy’nin kullanacağı dil modelini yapılandırmak için `dspy.LM` sınıfını ve `dspy.configure()` yöntemini kullanacağız.

```python
from typing import List

class PeopleExtraction(dspy.Signature):
    """
    Varsa, string token listesi içinden belirli kişilere atıfta bulunan bitişik token'ları çıkar.
    Çıktı olarak bir token listesi ver. Başka bir deyişle, birden fazla token'ı tek bir değerde birleştirme.
    """
    tokens: list[str] = dspy.InputField(desc="token'laştırılmış metin")
    extracted_people: list[str] = dspy.OutputField(desc="token'laştırılmış metinden çıkarılan, belirli kişilere atıfta bulunan tüm token'lar")

people_extractor = dspy.ChainOfThought(PeopleExtraction)
```

Burada DSPy’ye programımızda OpenAI’ın `gpt-4o-mini` modelini kullanmasını söylüyoruz. Kimlik doğrulama için DSPy, `OPENAI_API_KEY` anahtarınızı okur. Bunu kolayca [başka sağlayıcılarla veya yerel modellerle](https://github.com/stanfordnlp/dspy/blob/main/examples/migration.ipynb) değiştirebilirsiniz.

```python
lm = dspy.LM(model="openai/gpt-4o-mini")
dspy.configure(lm=lm)
```

## Metrik ve Değerlendirme Fonksiyonlarını Tanımlama

DSPy’de bir programın performansını değerlendirmek, yinelemeli geliştirme için kritik öneme sahiptir. İyi bir değerlendirme çerçevesi bize şunları sağlar:
- Programımızın çıktılarının kalitesini ölçmek.
- Çıktıları gerçek etiketlerle karşılaştırmak.
- İyileştirme alanlarını belirlemek.

**Ne Yapacağız:**
- Çıkarılan varlıkların gerçek değerlerle eşleşip eşleşmediğini değerlendirmek için özel bir metrik (`extraction_correctness_metric`) tanımlayacağız.
- Bu metriği bir eğitim veya test veri kümesine uygulayarak genel doğruluğu hesaplamak için bir değerlendirme fonksiyonu (`evaluate_correctness`) oluşturacağız.

Değerlendirme fonksiyonu, paralellik ve sonuçların görselleştirilmesini yönetmek için DSPy’nin `Evaluate` yardımcı aracını kullanır.

```python
def extraction_correctness_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> bool:
    """
    Varlık çıkarma tahminlerinin doğruluğunu hesaplar.
    
    Args:
        example (dspy.Example): Beklenen kişi varlıklarını içeren veri kümesi örneği.
        prediction (dspy.Prediction): DSPy kişi çıkarma programından gelen tahmin.
        trace: Hata ayıklama için isteğe bağlı iz nesnesi.
    
    Returns:
        bool: Tahminler beklentilerle eşleşiyorsa True, aksi halde False.
    """
    return prediction.extracted_people == example.expected_extracted_people

evaluate_correctness = dspy.Evaluate(
    devset=test_set,
    metric=extraction_correctness_metric,
    num_threads=24,
    display_progress=True,
    display_table=True
)
```

## İlk Çıkarıcıyı Değerlendirme

Programımızı optimize etmeden önce, mevcut performansını anlamak için bir başlangıç değerlendirmesine ihtiyacımız var. Bu bize şunları sağlar:
- Optimizasyondan sonra karşılaştırma yapmak için bir referans noktası oluşturmak.
- İlk uygulamadaki potansiyel zayıflıkları belirlemek.

Bu adımda, `people_extractor` programımızı test kümesi üzerinde çalıştıracak ve doğruluğunu daha önce tanımlanan değerlendirme çerçevesiyle ölçeceğiz.

```python
evaluate_correctness(people_extractor, devset=test_set)
```

<details>
<summary>MLflow Experiment içinde Değerlendirme Sonuçlarını İzleme</summary>

<br/>

Değerlendirme sonuçlarını zaman içinde takip etmek ve görselleştirmek için sonuçları MLflow Experiment içine kaydedebilirsiniz.


```python
import mlflow

with mlflow.start_run(run_name="extractor_evaluation"):
    evaluate_correctness = dspy.Evaluate(
        devset=test_set,
        metric=extraction_correctness_metric,
        num_threads=24,
        display_progress=True,
    )

    # Programı her zamanki gibi değerlendir
    result = evaluate_correctness(people_extractor)

    # Toplu skoru kaydet
    mlflow.log_metric("exact_match", result.score)
    # Ayrıntılı değerlendirme sonuçlarını tablo olarak kaydet
    mlflow.log_table(
        {
            "Tokens": [example.tokens for example in test_set],
            "Expected": [example.expected_extracted_people for example in test_set],
            "Predicted": [output[1] for output in result.results],
            "Exact match": [output[2] for output in result.results],
        },
        artifact_file="eval_results.json",
    )
```

Entegrasyon hakkında daha fazla bilgi edinmek için [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) sayfasını da ziyaret edin.

</details>

## Modeli Optimize Etme

DSPy, sisteminizin kalitesini artırabilecek güçlü optimizasyon araçları içerir.

Burada, DSPy’nin `MIPROv2` optimize edicisini kullanarak şunları yapıyoruz:
- Programın dil modeli (LM) istemini otomatik olarak ayarlamak: 1. istemin talimatlarını ayarlamak için LM’i kullanarak ve 2. `dspy.ChainOfThought` tarafından üretilen akıl yürütmeyle zenginleştirilmiş az örnekli örnekler oluşturarak.
- Eğitim kümesi üzerinde doğruluğu en üst düzeye çıkarmak.

Bu optimizasyon süreci otomatikleştirilmiştir; böylece doğruluğu artırırken zaman ve emek tasarrufu sağlar.

```python
mipro_optimizer = dspy.MIPROv2(
    metric=extraction_correctness_metric,
    auto="medium",
)
optimized_people_extractor = mipro_optimizer.compile(
    people_extractor,
    trainset=train_set,
    max_bootstrapped_demos=4,
    minibatch=False
)
```

## Optimize Edilmiş Programı Değerlendirme

Optimizasyondan sonra, iyileşmeleri ölçmek için programı test kümesi üzerinde yeniden değerlendiriyoruz. Optimize edilmiş ve ilk sonuçları karşılaştırmak bize şunları sağlar:
- Optimizasyonun faydalarını nicel olarak görmek.
- Programın görülmemiş verilere iyi genellendiğini doğrulamak.

Bu durumda, programın test veri kümesindeki doğruluğunun önemli ölçüde arttığını görüyoruz.

```python
evaluate_correctness(optimized_people_extractor, devset=test_set)
```

## Optimize Edilmiş Programın İstemini İnceleme

Programı optimize ettikten sonra, DSPy’nin programın istemini az örnekli örneklerle nasıl zenginleştirdiğini görmek için etkileşim geçmişini inceleyebiliriz. Bu adım şunları gösterir:
- Program tarafından kullanılan istemin yapısını.
- Modele rehberlik etmek için az örnekli örneklerin nasıl eklendiğini.

Son etkileşimi görüntülemek ve üretilen istemi analiz etmek için `inspect_history(n=1)` kullanın.

```python
dspy.inspect_history(n=1)
```

## Maliyeti göz önünde bulundurma

DSPy, programlarınızın maliyetini takip etmenize olanak tanır. Aşağıdaki kod, DSPy çıkarıcı programı tarafından şimdiye kadar yapılan tüm LM çağrılarının maliyetinin nasıl elde edileceğini gösterir.

```python
cost = sum([x['cost'] for x in lm.history if x['cost'] is not None])  # belirli sağlayıcılarda LiteLLM tarafından hesaplanan USD cinsinden maliyet
cost
```

## Optimize Edilmiş Programları Kaydetme ve Yükleme

DSPy, programları kaydetmeyi ve yüklemeyi destekler; böylece optimize edilmiş sistemleri baştan yeniden optimize etmeye gerek kalmadan yeniden kullanabilirsiniz. Bu özellik, programlarınızı üretim ortamlarında devreye almak veya iş arkadaşlarınızla paylaşmak için özellikle faydalıdır.

Bu adımda, optimize edilmiş programı bir dosyaya kaydedeceğiz ve gelecekte kullanmak üzere nasıl tekrar yükleneceğini göstereceğiz.

```python
optimized_people_extractor.save("optimized_extractor.json")

loaded_people_extractor = dspy.ChainOfThought(PeopleExtraction)
loaded_people_extractor.load("optimized_extractor.json")

loaded_people_extractor(tokens=["Italy", "recalled", "Marcello", "Cuttitta"]).extracted_people
```

<details>
<summary>Programları MLflow Experiment içinde kaydetme</summary>

<br/>

Programı yerel bir dosyaya kaydetmek yerine, daha iyi yeniden üretilebilirlik ve iş birliği için MLflow içinde takip edebilirsiniz.

1. **Bağımlılık Yönetimi**: MLflow, yeniden üretilebilirliği sağlamak için dondurulmuş ortam meta verisini programla birlikte otomatik olarak kaydeder.
2. **Deney Takibi**: MLflow ile programın performansını ve maliyetini, programın kendisiyle birlikte takip edebilirsiniz.
3. **İş Birliği**: MLflow deneyini paylaşarak programı ve sonuçları ekip üyelerinizle paylaşabilirsiniz.

Programı MLflow içine kaydetmek için aşağıdaki kodu çalıştırın:

```python
import mlflow

# Bir MLflow Run başlatın ve programı kaydedin
with mlflow.start_run(run_name="optimized_extractor"):
    model_info = mlflow.dspy.log_model(
        optimized_people_extractor,
        artifact_path="model", # Programı MLflow içinde kaydetmek için herhangi bir ad
    )

# Programı MLflow'dan tekrar yükleyin
loaded = mlflow.dspy.load_model(model_info.model_uri)
```

Entegrasyon hakkında daha fazla bilgi edinmek için [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) sayfasını da ziyaret edin.

</details>

## Sonuç

Bu eğitimde şunları gösterdik:
- Varlık çıkarma için modüler ve yorumlanabilir bir sistem kurmak amacıyla DSPy’yi kullanmayı.
- DSPy’nin yerleşik araçlarını kullanarak sistemi değerlendirmeyi ve optimize etmeyi.

Yapılandırılmış girdiler ve çıktılardan yararlanarak sistemin anlaşılmasını ve iyileştirilmesini kolaylaştırdık. Optimizasyon süreci, istemleri elle yazmadan veya parametrelerle tek tek oynamadan performansı hızlıca artırmamıza olanak sağladı.

**Sonraki Adımlar:**
- Başka varlık türlerinin çıkarımını deneyin (ör. konumlar veya kurumlar).
- Daha karmaşık akıl yürütme görevleri için `ReAct` gibi DSPy’nin diğer yerleşik modüllerini keşfedin.
- Sistemi büyük ölçekli belge işleme veya özetleme gibi daha büyük iş akışlarında kullanın.
