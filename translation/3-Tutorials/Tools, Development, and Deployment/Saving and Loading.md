# Eğitim: DSPy programınızı kaydetme ve yükleme

Bu kılavuz, DSPy programınızı nasıl kaydedip yükleyeceğinizi gösterir. Genel olarak, DSPy programınızı kaydetmenin iki yolu vardır:

1. Yalnızca programın durumunu kaydetmek; bu, PyTorch’ta yalnızca ağırlıkları kaydetmeye benzer.
2. Hem mimariyi hem de durumu içeren tüm programı kaydetmek; bu özellik `dspy>=2.6.0` ile desteklenir.

## Yalnızca Durumu Kaydetme

Durum, DSPy programının iç durumunu temsil eder; buna imza, demolar (few-shot örnekleri) ve program içindeki her `dspy.Predict` için kullanılacak `lm` gibi bilgiler dahildir. Ayrıca `dspy.retrievers.Retriever` için `k` gibi diğer DSPy modüllerinin yapılandırılabilir özniteliklerini de içerir. Bir programın durumunu kaydetmek için `save` metodunu kullanın ve `save_program=False` olarak ayarlayın. Durumu bir JSON dosyasına ya da pickle dosyasına kaydetmeyi seçebilirsiniz. JSON dosyasına kaydetmenizi öneririz; çünkü daha güvenlidir ve okunabilir. Ancak bazen programınız `dspy.Image` veya `datetime.datetime` gibi serileştirilemeyen nesneler içerebilir; bu durumda durumu bir pickle dosyasına kaydetmelisiniz. fileciteturn21file0

Diyelim ki bir programı bazı verilerle derledik ve gelecekte kullanmak üzere kaydetmek istiyoruz: fileciteturn21file0

```python
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

gsm8k = GSM8K()
gsm8k_trainset = gsm8k.train[:10]
dspy_program = dspy.ChainOfThought("question -> answer")

optimizer = dspy.BootstrapFewShot(metric=gsm8k_metric, max_bootstrapped_demos=4, max_labeled_demos=4, max_rounds=5)
compiled_dspy_program = optimizer.compile(dspy_program, trainset=gsm8k_trainset)
```

Programınızın durumunu bir JSON dosyasına kaydetmek için: fileciteturn21file0

```python
compiled_dspy_program.save("./dspy_program/program.json", save_program=False)
```

Programınızın durumunu bir pickle dosyasına kaydetmek için: fileciteturn21file0

!!! danger "Güvenlik Uyarısı: Pickle Dosyaları Rastgele Kod Çalıştırabilir"
    `.pkl` dosyalarının yüklenmesi rastgele kod çalıştırabilir ve tehlikeli olabilir. Pickle dosyalarını yalnızca güvenilir kaynaklardan ve güvenli ortamlarda yükleyin. **Mümkün olduğunda `.json` dosyalarını tercih edin**. Pickle kullanmanız gerekiyorsa, kaynağa güvendiğinizden emin olun ve yüklerken `allow_pickle=True` parametresini kullanın.

```python
compiled_dspy_program.save("./dspy_program/program.pkl", save_program=False)
```

Kaydedilmiş durumunuzu yüklemek için önce **aynı programı yeniden oluşturmanız**, ardından `load` metodunu kullanarak durumu yüklemeniz gerekir. fileciteturn21file0

```python
loaded_dspy_program = dspy.ChainOfThought("question -> answer") # Aynı programı yeniden oluştur.
loaded_dspy_program.load("./dspy_program/program.json")

assert len(compiled_dspy_program.demos) == len(loaded_dspy_program.demos)
for original_demo, loaded_demo in zip(compiled_dspy_program.demos, loaded_dspy_program.demos):
    # Yüklenen demo bir dict'tir; orijinal demo ise bir dspy.Example nesnesidir.
    assert original_demo.toDict() == loaded_demo
assert str(compiled_dspy_program.signature) == str(loaded_dspy_program.signature)
```

Ya da durumu bir pickle dosyasından yükleyebilirsiniz: fileciteturn21file0

!!! danger "Güvenlik Uyarısı"
    Pickle dosyalarını yüklerken `allow_pickle=True` kullanmayı unutmayın ve yalnızca güvenilir kaynaklardan yükleyin.

```python
loaded_dspy_program = dspy.ChainOfThought("question -> answer") # Aynı programı yeniden oluştur.
loaded_dspy_program.load("./dspy_program/program.pkl", allow_pickle=True)

assert len(compiled_dspy_program.demos) == len(loaded_dspy_program.demos)
for original_demo, loaded_demo in zip(compiled_dspy_program.demos, loaded_dspy_program.demos):
    # Yüklenen demo bir dict'tir; orijinal demo ise bir dspy.Example nesnesidir.
    assert original_demo.toDict() == loaded_demo
assert str(compiled_dspy_program.signature) == str(loaded_dspy_program.signature)
```

## Tüm Programı Kaydetme

!!! warning "Güvenlik Bildirimi: Tüm Programı Kaydetme İşlemi Pickle Kullanır"
    Tüm programı kaydetme işlemi serileştirme için `cloudpickle` kullanır; bu da pickle dosyalarıyla aynı güvenlik risklerine sahiptir. Programları yalnızca güvenilir kaynaklardan ve güvenli ortamlarda yükleyin.

`dspy>=2.6.0` sürümünden itibaren DSPy, hem mimariyi hem de durumu içeren tüm programın kaydedilmesini destekler. Bu özellik, Python nesnelerini serileştirme ve geri yükleme için kullanılan `cloudpickle` kütüphanesi tarafından sağlanır. fileciteturn21file0

Tüm programı kaydetmek için `save` metodunu kullanın, `save_program=True` olarak ayarlayın ve dosya adı yerine programın kaydedileceği bir dizin yolu belirtin. Dizin yolu istememizin nedeni, programın kendisiyle birlikte bağımlılık sürümleri gibi bazı meta verileri de kaydetmemizdir. fileciteturn21file0

```python
compiled_dspy_program.save("./dspy_program/", save_program=True)
```

Kaydedilmiş programı yüklemek için doğrudan `dspy.load` metodunu kullanın: fileciteturn21file0

```python
loaded_dspy_program = dspy.load("./dspy_program/")

assert len(compiled_dspy_program.demos) == len(loaded_dspy_program.demos)
for original_demo, loaded_demo in zip(compiled_dspy_program.demos, loaded_dspy_program.demos):
    # Yüklenen demo bir dict'tir; orijinal demo ise bir dspy.Example nesnesidir.
    assert original_demo.toDict() == loaded_demo
assert str(compiled_dspy_program.signature) == str(loaded_dspy_program.signature)
```

Tüm programı kaydetme yönteminde programı yeniden oluşturmanız gerekmez; mimariyi durumla birlikte doğrudan yükleyebilirsiniz. İhtiyaçlarınıza göre uygun kaydetme yaklaşımını seçebilirsiniz. fileciteturn21file0

### İçe Aktarılan Modülleri Serileştirme

`save_program=True` ile bir program kaydederken, programınızın bağlı olduğu özel modülleri de dahil etmeniz gerekebilir. Bu, programınız bu modüllere bağlıysa fakat yükleme sırasında `dspy.load` çağrılmadan önce bu modüller içe aktarılmıyorsa gereklidir. fileciteturn21file0

Programınızı kaydederken `save` metodunu çağırırken `modules_to_serialize` parametresine bu özel modülleri geçirerek hangi modüllerin programla birlikte serileştirileceğini belirtebilirsiniz. Bu, programınızın dayandığı bağımlılıkların serileştirme sırasında dahil edilmesini ve daha sonra program yüklenirken kullanılabilir olmasını sağlar. fileciteturn21file0

Arka planda bu işlem, bir modülü “değere göre pickle edilebilir” olarak kaydetmek için cloudpickle’ın `cloudpickle.register_pickle_by_value` fonksiyonunu kullanır. Bir modül bu şekilde kaydedildiğinde, cloudpickle modülü referansla değil değerle serileştirir; böylece modül içeriği kaydedilen programla birlikte korunur. fileciteturn21file0

Örneğin programınız özel modüller kullanıyorsa: fileciteturn21file0

```python
import dspy
import my_custom_module

compiled_dspy_program = dspy.ChainOfThought(my_custom_module.custom_signature)

# Programı özel modülle birlikte kaydet
compiled_dspy_program.save(
    "./dspy_program/",
    save_program=True,
    modules_to_serialize=[my_custom_module]
)
```

Bu, gerekli modüllerin doğru şekilde serileştirilmesini ve program daha sonra yüklenirken kullanılabilir olmasını sağlar. `modules_to_serialize` parametresine istediğiniz sayıda modül geçebilirsiniz. `modules_to_serialize` belirtmezseniz, serileştirme için ek bir modül kaydedilmez. fileciteturn21file0

## Geriye Dönük Uyumluluk

`dspy<3.0.0` itibarıyla kaydedilmiş programların geriye dönük uyumluluğunu garanti etmiyoruz. Örneğin, programı `dspy==2.5.35` ile kaydettiyseniz, yükleme sırasında da aynı DSPy sürümünü kullandığınızdan emin olun; aksi halde program beklendiği gibi çalışmayabilir. Farklı bir DSPy sürümünde kaydedilmiş bir dosyanın yüklenmesi hata vermeyebilir; ancak performans, programın kaydedildiği zamankinden farklı olabilir. fileciteturn21file0

`dspy>=3.0.0` sürümünden itibaren, büyük sürümler içinde kaydedilmiş programların geriye dönük uyumluluğunu garanti edeceğiz; yani `dspy==3.0.0` ile kaydedilmiş programlar `dspy==3.7.10` içinde yüklenebilir olmalıdır. fileciteturn21file0
