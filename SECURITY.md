# Güvenlik ve gizlilik notları

## Proje durumu

Bu depo yerel portföy MVP'sidir. FastAPI ve Streamlit ile yerel PDF yükleme, embedding, arama ve Ollama RAG akışı sunar. Kimlik doğrulaması ve canlı dağıtım yoktur. Proje tam güvenlik denetiminden geçmemiştir ve üretime hazır olarak tanımlanmamalıdır.

## Gizli bilgiler

- Gerçek anahtarları `.env` içinde yerel olarak saklayın; `.gitignore` `.env` dosyasını dışarıda bırakır.
- `.env.example` örnek değerler içindir; içine gerçek anahtar, parola veya kullanıcı verisi koymayın.
- Anahtarları kaynak koda, test fixture'larına, issue'lara veya ekran görüntülerine eklemeyin.
- `OPENAI_API_KEY` ve `OPENAI_MODEL` mevcut yerel RAG akışında kullanılmaz. LLM yanıtları için gerçek API anahtarı gerekmez.
- Bir anahtar yanlışlıkla Git'e gönderildiyse yalnızca dosyadan silmek yeterli değildir; sağlayıcıdan iptal edip yenileyin ve Git geçmişini temizlemeyi değerlendirin.

## Yerel işleme ve harici servisler

PDF metin çıkarımı, chunking, embedding çıkarımı ve semantik arama yerel olarak çalışır. RAG kullanıldığında soru ile seçilmiş belge parçaları varsayılan olarak `127.0.0.1:11434` adresindeki yerel Ollama API'sine gönderilir. Model ağırlıklarının ilk indirilmesi için internet gerekir; embedding modeli yönergelerle elle indirilir ve API otomatik indirme yapmaz. `OLLAMA_BASE_URL` varsayılan loopback adresinden uzağa ayarlanırsa uzak hosta erişim ayrıca `OLLAMA_ALLOW_REMOTE=true` ile açılmalıdır. Böyle bir ayar soru ve seçilmiş belge metinlerinin o makineye gönderilmesine yol açar. Bu projede ücretli LLM entegrasyonu veya otomatik ücretli yedek yol yoktur.

Proje geliştirme hedefi 0 TL'dir. Kullanılan açık kaynak paketleri ve yerel çalışma seçeneği kendi başına API faturası oluşturmaz; yerel model çalıştırmak için uyumlu donanım, disk ve ilk ağırlık indirmesi gerekebilir. Ücretsiz hosting veya API kotaları değişebilir; gelecekte değerlendirildiklerinde güncel koşullar ayrıca incelenmelidir. Ücretsiz bir servis hiçbir zaman doğrulamadan ücretsiz kabul edilmemelidir.

## Mevcut PDF işleme korumaları

- Yapılandırılabilir dosya boyutu ve sayfa sayısı sınırları uygulanır.
- Dosya yolu girdisi yalnızca yerel servis kullanımı içindir. İstemcinin gönderdiği dosya adı gösterim metadata'sıdır; dosya yolu olarak açılmaz ve yol bileşenleri atılır.
- PDF bytes boyut sınırına göre okunur ve indeksleme sonrası uygulama kalıcı PDF kopyası tutmaz. Multipart parser büyük yükleri request ömrü boyunca geçici olarak spool edebilir.
- Boş, bozuk, şifreli ve sınırı aşan dosyalar kontrollü `PdfProcessingError` ile reddedilir.
- Kod belge metnini, prompt'ları veya gizli değerleri loglamaz. Embedding servisi ve varsayılan Ollama adapter'ı metni yerel modele verir. Gelecekte log eklenirse ham PDF, çıkarılan metin, soru ve anahtarlar varsayılan olarak loglanmamalıdır.
- PDF parser'ları karmaşık ve saldırgan girdilerle karşılaşabilir. Boyut ve sayfa sınırı tek başına decompression bomb veya parser açığını önlemez; halka açık yükleme açılmadan önce güncel bağımlılık takibi, ek kaynak sınırları ve izolasyon gerekir.

## Demo ve çok kullanıcılı kullanım için yapılacaklar

Mevcut MVP PDF uzantısını ve içeriğini doğrular, boyut/sayfa/chunk sınırları uygular ve yüklenen belgeyi kalıcı dosyaya yazmaz. Halka açık bir demo ise kimlik doğrulama, istek oranı sınırı, kötü amaçlı PDF izolasyonu ve gözden geçirilmiş depolama yaşam döngüsü eklenmeden açılmamalıdır. Gerçek kullanıcılardan gizli belge veya kişisel veri yüklemeleri istenmemelidir.

API her oturum için ayrı `InMemoryVectorIndex` kullanır; rastgele, yüksek entropili oturum kimliğiyle indeks erişimi yapılır, oturum silme ve 60 dakika etkin olmama TTL'i indeks verisini bellekten temizler. Bu yerel MVP'de oturum kimliği pratikte erişim anahtarı gibi davranır ve uygulama kimlik doğrulaması sağlamaz. Varsayılan Uvicorn bind adresi `127.0.0.1` olmalıdır. Halka açık demo için kimlik doğrulama veya sıkı erişim sınırları olmadan kişisel belge yüklemesi açılmamalıdır.

PDF metni LLM bağlamına girerken JSON alanlarında taşınır ve system prompt'ta güvenilmeyen veri olarak belirtilir. Belge talimatları sistem talimatlarının yerini alamaz. Bu yönergeler prompt injection riskini tamamen ortadan kaldırmaz. RAG yanıtındaki kaynak kimlikleri yalnızca o isteğin retrieval sonuçlarıyla eşleştirilir; bilinmeyen kimlikler cevap kaynağı metadata'sı olarak gösterilmez. Modelin normal metin içindeki her iddiasının gerçekten desteklendiği otomatik olarak kanıtlanamaz; daha kapsamlı değerlendirme gerekir.

Streamlit, kullanıcı ve model metnini HTML olarak doğrudan çalıştırmaz. Hata yanıtları iç yolları, prompt'ları, provider yanıt gövdelerini veya anahtarları sızdırmamalıdır. API anahtarları yalnızca sunucu tarafında saklanmalıdır.

## Sağlayıcıdan bağımsız LLM mimarisi

RAG servisi küçük bir `AnswerGenerator` Python `Protocol` arkasından yanıt üretir; mevcut adapter standart kütüphane HTTP istemcisi kullanan Ollama istemcisidir. HTTP ve Ollama ayrıntıları RAG akışına karışmaz. Sağlayıcı değiştirilirse yalnızca yeni adapter ve ona ait testler eklenmelidir. Kullanılmayan adapter'lar veya genel plugin sistemi kurulmamalıdır.

Ollama üretim modeli `OLLAMA_MODEL`, embedding modeli `EMBEDDING_MODEL` ile bağımsız yapılandırılır. Varsayılan model `qwen3:1.7b`'dir. Ücretli servise otomatik fallback yapılmaz. Embedding modeli veya boyutu değişirse mevcut vektör indeksinin yeniden oluşturulması gerekebilir.

Gelecekte sağlayıcı/model değerlendirmesi için küçük, sürümlenmiş ve hassas veri içermeyen bir set hazırlanmalıdır: sentetik Türkçe kaynak parçaları, sorular, beklenen cevap kapsamı, destekleyen kaynak kimlikleri ve cevap verilememesi gereken sorular. Her modelde aynı set çalıştırılarak Türkçe yanıt kalitesi, desteklenmeyen iddia/abstention, kaynak tutarlılığı, çıktı şeması ve hata davranışı ölçülmelidir. Aşama 5 offline testleri sahte modelle çalışır; 2026-10-23'teki önceki opt-in Ollama entegrasyon testi yalnızca sentetik bir örnekte yanıt ve kaynak zincirini doğrulamıştır, genel yanıt kalitesini ölçmemiştir.

## Denetim kapsamı ve raporlama

Bu Aşama 5 çalışmasında gerçek veya kişisel PDF kullanılmadı, Ollama testi tekrar çalıştırılmadı, ücretli servise istek yapılmadı ve deployment denenmedi. Önceki Aşama 4.5 entegrasyon testi 2026-10-23'te sentetik belge ve yerel `qwen3:1.7b` ile başarılı olmuştu. Çalışma klasöründe Git deposu bulunmadığından commit geçmişi veya Git tarafından izlenme durumu doğrulanamadı. `.gitignore` içinde `.env` kuralı vardır; `.env` dosyasının içeriği okunmamıştır. Gelecekte Git deposu oluşturulduğunda gizli bilgi taraması ve bağımlılık güvenlik denetimi ayrıca yapılmalıdır.

Ruff mevcut statik lint kontrolüdür. Proje servis ve model katmanları büyüdüğünde mypy veya Pyright ile tip denetimi faydalı olur; şimdilik ek bağımlılık ve yapılandırma maliyetini haklı çıkaracak kapsam oluşmamıştır. Bağımlılık açıklarını düzenli kontrol etmek için ileride `pip-audit` gibi açık kaynaklı bir denetim aracı CI veya yerel rutine eklenebilir; bu kontrol genellikle güncel advisory verisine erişmek için ağ ister. Bu aşamada bağımlılık güvenlik taraması çalıştırılmamıştır; Ruff güvenlik taraması yerine geçmez.
