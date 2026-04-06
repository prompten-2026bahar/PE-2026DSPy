---
sidebar_position: 998
---

!!! warning "Bu sayfa güncelliğini yitirmiştir ve DSPy 2.5 ile 2.6 sürümlerinde tamamen doğru olmayabilir"


# SSS

## DSPy benim için doğru araç mı? DSPy ve diğer çatılar

**DSPy** felsefesi ve soyutlama katmanı diğer kütüphane ve çatılarınkinden önemli ölçüde farklıdır; bu nedenle **DSPy**’nin sizin kullanım senaryonuz için doğru çatı olup olmadığına karar vermek genellikle oldukça kolaydır. Eğer bir NLP/AI araştırmacısıysanız (veya yeni iş akışlarını ya da yeni görevleri keşfeden bir uygulayıcıysanız), cevap çoğunlukla değişmez şekilde **evet** olacaktır. Eğer başka şeylerle uğraşan bir uygulayıcıysanız, okumaya devam edin.

**DSPy vs. istemler için ince sarmalayıcılar (OpenAI API, MiniChain, temel şablonlama)** Başka bir deyişle: _İstemlerimi neden doğrudan string template olarak yazmayayım?_ Çok basit senaryolarda bu _gayet iyi_ çalışabilir. (Sinir ağlarına aşinaysanız, bu küçük iki katmanlı bir yapay sinir ağını Python `for` döngüsüyle ifade etmeye benzer. Bir şekilde çalışır.) Ancak daha yüksek kaliteye (veya yönetilebilir maliyete) ihtiyaç duyduğunuzda, çok aşamalı ayrıştırmaları, gelişmiş istemleri, veri bootstrapping’i, dikkatli fine-tuning’i, retrieval augmentation’ı ve/veya daha küçük (ya da daha ucuz, ya da yerel) modelleri yinelemeli biçimde keşfetmeniz gerekir. Temel model tabanlı sistemler kurmanın gerçek ifade gücü, bu bileşenler arasındaki etkileşimlerde yatar. Ancak bir parçayı her değiştirdiğinizde, büyük olasılıkla diğer birkaç bileşeni de bozarsınız (veya zayıflatırsınız). **DSPy**, bu etkileşimlerin gerçek sistem tasarımınızın dışında kalan parçalarını temiz biçimde soyutlar (_ve_ güçlü biçimde optimize eder). Böylece modül düzeyindeki etkileşimleri tasarlamaya odaklanabilirsiniz: **DSPy** ile 10 veya 20 satırda ifade edilen _aynı program_, kolaylıkla `GPT-4` için çok aşamalı talimatlara, `Llama2-13b` için ayrıntılı istemlere veya `T5-base` için fine-tuning süreçlerine derlenebilir. Ayrıca artık projenizin merkezinde uzun, kırılgan ve modele özgü string’leri sürdürmek zorunda kalmazsınız.

**DSPy vs. LangChain, LlamaIndex gibi uygulama geliştirme kütüphaneleri** LangChain ve LlamaIndex, üst düzey uygulama geliştirmeyi hedefler; veriniz veya yapılandırmanızla takılabilen, _hazır pilli_ uygulama modülleri sunarlar. PDF’ler üzerinde soru-cevap ya da standart text-to-SQL gibi genel ve raftan alınmış istemleri kullanmaktan memnunsanız, bu kütüphanelerde zengin bir ekosistem bulursunuz. **DSPy** ise dahili olarak belirli uygulamaları hedefleyen el yapımı istemler içermez. Bunun yerine **DSPy**, çok daha güçlü ve genel amaçlı küçük bir modül kümesi sunar; bunlar, _veriniz üzerinde ve iş akışınız içinde LM’inize istem vermeyi (veya fine-tune etmeyi) öğrenebilir_. Verinizi değiştirdiğinizde, programın kontrol akışında düzenlemeler yaptığınızda veya hedef LM’i değiştirdiğinizde, **DSPy derleyicisi** programınızı bu iş akışı için özel olarak optimize edilmiş yeni istemler (veya fine-tuning süreçleri) kümesine eşleyebilir. Bu nedenle, kendi kısa programınızı yazmaya (veya genişletmeye) istekliyseniz, **DSPy**’nin göreviniz için en yüksek kaliteyi en az çabayla sağladığını görebilirsiniz. Kısacası, **DSPy**, önceden tanımlanmış istemler ve entegrasyonlar kütüphanesi değil; hafif ama otomatik optimize olan bir programlama modeli istediğinizde kullanılır. Sinir ağlarına aşinaysanız: Bu, PyTorch (yani **DSPy**’yi temsil eden) ile HuggingFace Transformers (yani üst düzey kütüphaneleri temsil eden) arasındaki farka benzer.

**DSPy vs. Guidance, LMQL, RELM, Outlines gibi üretim kontrolü kütüphaneleri** Bunların hepsi, örneğin JSON çıktı şeması zorlamak veya örneklemeyi belirli bir düzenli ifadeyle sınırlamak gibi, LM’lerin tekil tamamlamalarını kontrol etmeye yönelik heyecan verici yeni kütüphanelerdir. Bu pek çok durumda çok faydalıdır; ancak genel olarak tek bir LM çağrısının düşük seviyeli ve yapılandırılmış kontrolüne odaklanırlar. Elde ettiğiniz JSON’un (veya yapılandırılmış çıktının) göreviniz için gerçekten doğru veya yararlı olmasını sağlamazlar. Buna karşılık **DSPy**, programlarınızdaki istemleri çeşitli görev ihtiyaçlarına uyacak şekilde otomatik optimize eder; buna geçerli yapılandırılmış çıktı üretmek de dahil olabilir. Bununla birlikte, **DSPy** içindeki **Signatures** yapısının bu kütüphaneler tarafından uygulanabilecek regex benzeri kısıtları ifade etmesine izin vermeyi değerlendiriyoruz.

## Temel Kullanım

**Görevim için DSPy’yi nasıl kullanmalıyım?** Bunun için bir [sekiz adımlı rehber](learn/index.md) yazdık. Kısaca, DSPy kullanımı yinelemeli bir süreçtir. Önce görevinizi ve en üst düzeye çıkarmak istediğiniz metrikleri tanımlarsınız, ardından birkaç örnek girdi hazırlarsınız — tipik olarak etiketsiz (veya metrik gerektiriyorsa yalnızca nihai çıktılar için etiketli). Sonra yerleşik katmanlardan (`modules`) hangilerini kullanacağınızı seçerek iş akışınızı kurar, her katmana bir `signature` (girdi/çıktı tanımı) verirsiniz ve modüllerinizi Python kodunuz içinde serbestçe çağırırsınız. Son olarak, DSPy `optimizer` kullanarak kodunuzu yüksek kaliteli talimatlara, otomatik few-shot örneklere veya LM ağırlıklarına yönelik güncellemelere derlersiniz.

**Karmaşık istemimi nasıl DSPy iş akışına dönüştürürüm?** Yukarıdaki aynı yanıta bakın.

**DSPy optimize ediciler neyi ayarlar?** Ya da, _derleme yapmak aslında ne yapar?_ Her optimize edici farklıdır; ancak hepsi programınız üzerinde bir metriği en üst düzeye çıkarmak için istemleri veya LM ağırlıklarını güncellemeye çalışır. Mevcut DSPy `optimizers`; verinizi inceleyebilir, programınız boyunca izlekler simüle ederek her adım için iyi/kötü örnekler üretebilir, geçmiş sonuçlara göre her adım için talimatlar önerebilir ya da bunları iyileştirebilir, kendi ürettiği örnekler üzerinde LM ağırlıklarını fine-tune edebilir veya bunların birkaçını birleştirerek kaliteyi artırabilir ya da maliyeti azaltabilir. Daha zengin bir alanı keşfeden yeni optimize edicileri memnuniyetle birleştirmek isteriz: bugün prompt engineering, "synthetic data" üretimi veya self-improvement için manuel olarak yaptığınız pek çok adım muhtemelen, keyfi LM programları üzerinde çalışan bir DSPy optimize edicisine genellenebilir.

Diğer SSS’ler. Bunların her biri için buraya resmî yanıtlar ekleyecek PR’ları memnuniyetle karşılarız. Cevapların tümünü veya çoğunu mevcut issue’larda, eğitimlerde ya da makalelerde bulabilirsiniz.

- **Birden fazla çıktıyı nasıl alırım?**

Birden fazla çıktı alanı belirtebilirsiniz. Kısa imza biçiminde, `"->"` göstergesinden sonra birden fazla çıktıyı virgülle ayırarak listeleyebilirsiniz (ör. `"inputs -> output1, output2"`). Uzun imza biçiminde ise birden fazla `dspy.OutputField` tanımlayabilirsiniz.


- **Kendi metriklerimi nasıl tanımlarım? Metrikler float döndürebilir mi?**

Metrikleri, model üretimlerini işleyen ve kullanıcı tanımlı gereksinimlere göre değerlendiren basit Python fonksiyonları olarak tanımlayabilirsiniz. Metrikler mevcut veriyi (ör. gold label’lar) model tahminleriyle karşılaştırabilir veya çıktıların çeşitli bileşenlerini değerlendirmek için LM geri bildirimi (ör. LLM-as-Judge) kullanabilir. Metrikler `bool`, `int` ve `float` türünde skorlar döndürebilir. Özel metrik tanımlama ve AI geri bildirimi ve/veya DSPy programlarıyla gelişmiş değerlendirmeler hakkında daha fazla bilgi için resmî [Metrics docs](learn/evaluation/metrics.md) sayfasına göz atın.

- **Derleme ne kadar pahalı veya yavaş?**

Derleme metriklerini yansıtmak için, karşılaştırma amacıyla bir deney öne çıkarıyoruz: [BootstrapFewShotWithRandomSearch](api/optimizers/BootstrapFewShotWithRandomSearch.md) optimize edicisini `gpt-3.5-turbo-1106` modeli üzerinde, 7 aday program ve 10 thread ile kullanan bir program derleniyor. Bu programın derlenmesinin yaklaşık 6 dakika sürdüğü, 3200 API çağrısı yaptığı, 2.7 milyon giriş token’ı ve 156.000 çıkış token’ı kullandığı ve toplam maliyetin 3 ABD doları olduğu rapor edilmiştir (OpenAI modelinin mevcut fiyatlandırmasıyla).

DSPy `optimizers` derleme sırasında doğal olarak ek LM çağrılarına neden olur; ancak bu ek yükü, performansı en üst düzeye çıkarmayı amaçlayan minimal uygulamalarla gerekçelendiriyoruz. Bu da, derleme zamanında daha büyük modellerle DSPy programları derleyerek daha küçük modellerin performansını artırma ve bu gelişmiş davranışı çıkarım zamanında test edilen daha küçük modele aktarma yolları açar.  


## Dağıtım veya Yeniden Üretilebilirlik Kaygıları

- **Derlenmiş programımın bir checkpoint’ini nasıl kaydederim?**

İşte derlenmiş bir modülün kaydedilip yüklenmesine dair örnek:

```python
cot_compiled = teleprompter.compile(CoT(), trainset=trainset, valset=devset)

#Kaydetme
cot_compiled.save('compiled_cot_gsm8k.json')

#Yükleme:
cot = CoT()
cot.load('compiled_cot_gsm8k.json')
```

- **Dağıtım için nasıl dışa aktarırım?**

DSPy programlarını dışa aktarmak, yukarıda gösterildiği gibi onları kaydetmekten ibarettir!

- **Kendi verimde nasıl arama yaparım?**

[RAGatouille](https://github.com/bclavie/ragatouille) gibi açık kaynak kütüphaneler, ColBERT gibi gelişmiş retrieval modelleriyle kendi veriniz üzerinde arama yapmanıza, belgeleri gömmeye ve indekslemeye olanak tanır. DSPy programlarınızı geliştirirken aranabilir veri kümeleri oluşturmak için bu tür kütüphaneleri rahatlıkla entegre edebilirsiniz!

- **Önbelleği nasıl kapatırım? Önbelleği nasıl dışa aktarırım?**

v2.5’ten itibaren `dspy.LM` içinde `cache` parametresini `False` yaparak önbelleği kapatabilirsiniz:

```python
dspy.LM('openai/gpt-4o-mini',  cache=False)
```

Yerel önbelleğiniz, genel ortam dizini olan `os.environ["DSP_CACHEDIR"]` içine veya not defterleri için `os.environ["DSP_NOTEBOOK_CACHEDIR"]` içine kaydedilir. Genellikle önbellek dizinini `os.path.join(repo_path, 'cache')` olarak ayarlayıp buradan dışa aktarabilirsiniz:
```python
os.environ["DSP_NOTEBOOK_CACHEDIR"] = os.path.join(os.getcwd(), 'cache')
```

!!! warning "Önemli"
    `DSP_CACHEDIR`, eski istemcilerden (ör. dspy.OpenAI, dspy.ColBERTv2 vb.) sorumludur; `DSPY_CACHEDIR` ise yeni `dspy.LM` istemcisinden sorumludur.

    AWS lambda dağıtımında hem DSP_\* hem de DSPY_\* yapılarını devre dışı bırakmalısınız.


## İleri Düzey Kullanım

- **Nasıl paralelleştiririm?**
DSPy programlarını hem derleme hem de değerlendirme sırasında, ilgili DSPy `optimizers` içinde veya `dspy.Evaluate` yardımcı fonksiyonunda birden fazla thread ayarı belirterek paralelleştirebilirsiniz.

- **Bir modülü nasıl dondururum?**

Modüller, `._compiled` öznitelikleri `True` yapılarak dondurulabilir; bu, modülün optimize edici derlemesinden geçtiğini ve parametrelerinin artık ayarlanmaması gerektiğini gösterir. Bu durum, `dspy.BootstrapFewShot` gibi optimize ediciler içinde dahili olarak ele alınır; burada student program, teacher bootstrapping sürecinde toplanan few-shot gösterimlerini aktarmadan önce dondurulmuş olacak şekilde güvence altına alınır. 

- **DSPy assertions’ı nasıl kullanırım?**

    a) **Programa Assertion Nasıl Eklenir**:
    - **Kısıtlar Tanımlayın**: DSPy programınız içinde `dspy.Assert` ve/veya `dspy.Suggest` kullanarak kısıtlar tanımlayın. Bunlar, uygulamak istediğiniz sonuçlara yönelik boolean doğrulama kontrollerine dayanır ve model çıktılarınızı doğrulamak için basit Python fonksiyonları olabilir.
    - **Assertion’ları Entegre Etme**: Assertion ifadelerinizi model üretimlerinden sonra yerleştirin (ipucu: bir modül katmanından sonra)

    b) **Assertion’ları Nasıl Etkinleştirirsiniz**:
    1. **`assert_transform_module` Kullanımı**:
        - `assert_transform_module` fonksiyonunu ve bir `backtrack_handler` kullanarak DSPy modülünüzü assertion’larla sarın. Bu fonksiyon, programınızı dahili assertion backtracking ve retry mantığını içerecek şekilde dönüştürür; bu mantık özelleştirilebilir:
        `program_with_assertions = assert_transform_module(ProgramWithAssertions(), backtrack_handler)`
    2. **Assertion’ları Etkinleştirme**:
        - Assertion içeren DSPy programınız üzerinde doğrudan `activate_assertions` çağırın: `program_with_assertions = ProgramWithAssertions().activate_assertions()`

    **Not**: Assertion’ları doğru kullanmak için, `dspy.Assert` veya `dspy.Suggest` ifadeleri içeren bir DSPy programını yukarıdaki iki yöntemden biriyle mutlaka **etkinleştirmeniz** gerekir. 

## Hatalar

- **"context too long" hatalarıyla nasıl başa çıkarım?**

DSPy’de "context too long" hataları alıyorsanız, muhtemelen isteminize gösterimler eklemek için DSPy optimize edicileri kullanıyorsunuzdur ve bu mevcut bağlam pencerenizi aşıyordur. Bu parametreleri azaltmayı deneyin (ör. `max_bootstrapped_demos` ve `max_labeled_demos`). Ayrıca isteminizin model bağlam uzunluğuna sığmasını sağlamak için getirilen pasaj/belge/gömme sayısını da azaltabilirsiniz.

Daha genel bir çözüm ise LM isteğinde belirtilen `max_tokens` sayısını artırmaktır (ör. `lm = dspy.OpenAI(model = ..., max_tokens = ...`).

## Ayrıntı Düzeyini Ayarlama
DSPy, log basmak için [logging kütüphanesini](https://docs.python.org/3/library/logging.html) kullanır. DSPy kodunuzu hata ayıklamak istiyorsanız, aşağıdaki örnek kodla log seviyesini DEBUG olarak ayarlayın.

```python
import logging
logging.getLogger("dspy").setLevel(logging.DEBUG)
```

Alternatif olarak, log miktarını azaltmak istiyorsanız log seviyesini WARNING veya ERROR yapın.

```python
import logging
logging.getLogger("dspy").setLevel(logging.WARNING)
```