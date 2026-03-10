# DSPy'de Değerlendirme (Evaluation)

Başlangıç seviyesinde bir sistem kurduktan sonra, onu daha sistematik bir şekilde geliştirmek için **başlangıç aşamasında bir geliştirme seti (development set) toplama** zamanı gelmiş demektir. Görevinizle ilgili 20 adet girdi örneği bile faydalı olabilir, ancak 200 örnek sizi çok daha ileriye taşır. Belirlediğiniz *metriğe* bağlı olarak, ya sadece girdilere (hiç etiket olmadan) ya da hem girdilere hem de sisteminizin *nihai* çıktılarına ihtiyacınız olacaktır. (DSPy'de programınızdaki ara adımlar için etiketlere neredeyse hiçbir zaman ihtiyacınız olmaz.)

Görevinizle ilişkili veri setlerini HuggingFace datasets gibi platformlarda veya StackExchange gibi doğal kaynaklarda bulabilirsiniz. Lisansı uygun veriler varsa bunları kullanmanızı öneririz. Aksi takdirde, birkaç örneği elle etiketleyebilir veya sisteminizin bir demosunu yayına alarak başlangıç verilerini bu şekilde toplayabilirsiniz.


Ardından, **DSPy metriğinizi tanımlamalısınız**. Sisteminizden gelen çıktıları ne iyi veya kötü yapar? Metrikleri tanımlamaya ve bunları zamanla aşamalı olarak geliştirmeye yatırım yapın; tanımlayamadığınız bir şeyi tutarlı bir şekilde geliştirmek zordur. Metrik, verilerinizdeki örnekleri ve sisteminizin çıktısını alan ve karşılığında bir puan döndüren bir fonksiyondur.

Basit görevler (örneğin sınıflandırma veya kısa yanıtlı soru-cevap) için bu sadece "doğruluk" (accuracy) olabilir. Ancak çoğu uygulama için sisteminiz uzun formda çıktılar üretecektir; bu durumda metriğiniz, çıktının birden fazla özelliğini kontrol eden daha küçük bir DSPy programı olacaktır. Bunu ilk denemede doğru yapmak zordur: basit bir şeyle başlayın ve üzerinde yineleme (iterasyon) yapın.

Artık elinizde veri ve bir metrik olduğuna göre, tasarımınızın avantaj ve dezavantajlarını anlamak için boru hattınız (pipeline) üzerinde geliştirme değerlendirmeleri çalıştırın. Çıktılara ve metrik puanlarına bakın. Bu, muhtemelen büyük sorunları tespit etmenizi sağlayacak ve sonraki adımlarınız için bir temel (baseline) oluşturacaktır.

---

??? "Eğer metriğinizin kendisi bir DSPy programıysa..."
    Eğer metriğiniz bir DSPy programıysa, yineleme yapmanın güçlü bir yolu metriğin kendisini optimize etmektir. Metriğin çıktısı genellikle basit bir değer (örneğin 5 üzerinden bir puan) olduğundan, bunu yapmak genellikle kolaydır; metriğin kendi metriğini tanımlamak ve birkaç örnek toplayarak onu optimize etmek oldukça basittir.