# MLflow ile DSPy Optimize Edicilerini İzleme

Bu eğitim, DSPy optimizasyon sürecinizi izlemek ve analiz etmek için MLflow’u nasıl kullanacağınızı gösterir. MLflow’un DSPy için yerleşik entegrasyonu, DSPy optimizasyon deneyiminiz için izlenebilirlik ve hata ayıklanabilirlik sağlar. Optimizasyon sırasında oluşan ara denemeleri anlamanıza, optimize edilmiş programı ve sonuçlarını saklamanıza ve program yürütmeniz üzerinde gözlemlenebilirlik elde etmenize olanak tanır.

Autologging yeteneği sayesinde MLflow aşağıdaki bilgileri izler:

* **Optimize Edici Parametreleri**
    * Few-shot örnek sayısı
    * Aday sayısı
    * Diğer yapılandırma ayarları

* **Program Durumları**
    * Başlangıç talimatları ve few-shot örnekleri
    * Optimize edilmiş talimatlar ve few-shot örnekleri
    * Optimizasyon sırasında ortaya çıkan ara talimatlar ve few-shot örnekleri

* **Veri Kümeleri**
    * Kullanılan eğitim verisi
    * Kullanılan değerlendirme verisi

* **Performans İlerlemesi**
    * Genel metrik ilerleyişi
    * Her değerlendirme adımındaki performans

* **Trace’ler**
    * Program yürütme trace’leri
    * Model yanıtları
    * Ara istemler

## Başlarken

### 1. MLflow’u Kurun
Önce MLflow’u kurun (sürüm 2.21.1 veya sonrası):

```bash
pip install mlflow>=2.21.1
```

### 2. MLflow Tracking Sunucusunu Başlatın

Aşağıdaki komutla MLflow tracking sunucusunu başlatalım. Bu, `http://127.0.0.1:5000/` adresinde yerel bir sunucu başlatacaktır:

```bash
# MLflow tracing kullanırken SQL store kullanmanız şiddetle tavsiye edilir
mlflow server --backend-store-uri sqlite:///mydb.sqlite
```

### 3. Autologging’i Etkinleştirin

DSPy optimizasyonunuzu izlemek için MLflow’u yapılandırın:

```python
import mlflow
import dspy

# Tüm özelliklerle autologging'i etkinleştir
mlflow.dspy.autolog(
    log_compiles=True,    # Optimizasyon sürecini izle
    log_evals=True,       # Değerlendirme sonuçlarını izle
    log_traces_from_compile=True  # Optimizasyon sırasında program trace'lerini izle
)

# MLflow tracking'i yapılandır
mlflow.set_tracking_uri("http://localhost:5000")  # Yerel MLflow sunucusunu kullan
mlflow.set_experiment("DSPy-Optimization")
```

### 4. Programınızı Optimize Etme

Aşağıda, bir matematik problem çözücüsünün optimizasyonunu nasıl izleyeceğinizi gösteren tam bir örnek bulunmaktadır:

```python
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

# Dil modelinizi yapılandırın
lm = dspy.LM(model="openai/gpt-4o")
dspy.configure(lm=lm)

# Veri kümesini yükleyin
gsm8k = GSM8K()
trainset, devset = gsm8k.train, gsm8k.dev

# Programınızı tanımlayın
program = dspy.ChainOfThought("question -> answer")

# İzleme ile birlikte optimize ediciyi oluşturun ve çalıştırın
teleprompter = dspy.teleprompt.MIPROv2(
    metric=gsm8k_metric,
    auto="light",
)

# Optimizasyon süreci otomatik olarak izlenecektir
optimized_program = teleprompter.compile(
    program,
    trainset=trainset,
)
```

### 5. Sonuçları Görüntüleme

Optimizasyon tamamlandıktan sonra, sonuçları MLflow’un arayüzü üzerinden analiz edebilirsiniz. Optimizasyon çalıştırmalarınızı nasıl inceleyeceğinizi adım adım ele alalım.

#### Adım 1: MLflow Arayüzüne Erişin
MLflow tracking sunucusu arayüzüne erişmek için web tarayıcınızda `http://localhost:5000` adresine gidin.

#### Adım 2: Deney Yapısını Anlama
Deney sayfasını açtığınızda, optimizasyon sürecinizin hiyerarşik bir görünümünü göreceksiniz. Üst çalışma (parent run) genel optimizasyon sürecinizi temsil ederken, alt çalışmalar (child runs) optimizasyon sırasında oluşturulan programınızın her bir ara sürümünü gösterir.

![Experiments](./experiment.png)

#### Adım 3: Parent Run’ı Analiz Etme
Parent run’a tıkladığınızda optimizasyon sürecinizin büyük resmini görürsünüz. Optimize edicinizin yapılandırma parametreleri ve değerlendirme metriklerinizin zaman içinde nasıl ilerlediği hakkında ayrıntılı bilgilere ulaşırsınız. Parent run ayrıca son optimize edilmiş programınızı; kullanılan talimatlar, imza tanımları ve few-shot örnekleriyle birlikte saklar. Buna ek olarak, optimizasyon sürecinde kullanılan eğitim verisini de inceleyebilirsiniz.

![Parent Run](./parent_run.png)

#### Adım 4: Child Run’ları İnceleme
Her child run, belirli bir optimizasyon denemesinin ayrıntılı bir anlık görüntüsünü sunar. Deney sayfasından bir child run seçtiğinizde, o ara programa ait çeşitli yönleri inceleyebilirsiniz.
Run parameter sekmesinde veya artifact sekmesinde, ara program için kullanılan talimatları ve few-shot örneklerini gözden geçirebilirsiniz.
En güçlü özelliklerden biri Traces sekmesidir; bu sekme programınızın yürütülmesini adım adım gösterir. Burada DSPy programınızın girdileri tam olarak nasıl işlediğini ve çıktıları nasıl ürettiğini anlayabilirsiniz.

![Child Run](./child_run.png)

### 6. Çıkarım için Modelleri Yükleme
Optimize edilmiş programı, çıkarım için doğrudan MLflow tracking sunucusundan yükleyebilirsiniz:

```python
model_path = mlflow.artifacts.download_artifacts("mlflow-artifacts:/path/to/best_model.json")
program.load(model_path)
```

## Sorun Giderme

- Trace’ler görünmüyorsa `log_traces_from_compile=True` ayarının etkin olduğundan emin olun
- Büyük veri kümelerinde bellek sorunlarını önlemek için `log_traces_from_compile=False` ayarını değerlendirin
- MLflow çalışma verilerine programatik olarak erişmek için `mlflow.get_run(run_id)` kullanın

Daha fazla özellik için [MLflow Documentation](https://mlflow.org/docs/latest/llms/dspy) sayfasını inceleyin.
