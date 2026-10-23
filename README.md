# Türkçe Doküman AI Asistanı

PDF belgelerinize Türkçe soru sorun ve yanıtları sayfa kaynaklarıyla inceleyin. Bu proje; PDF metin çıkarımı, yerel embedding ve Ollama ile kaynaklı RAG akışı sunan, yerel kullanım için tasarlanmış bir portföy MVP'sidir.

> **Gizlilik:** PDF metni ve indeks FastAPI oturum belleğinde tutulur. Soru ile seçilen metin parçaları, yanıt üretimi için varsayılan olarak aynı bilgisayardaki Ollama'ya gönderilir. Belgeler yeniden başlatma sonrasında saklanmaz. Uygulama kimlik doğrulaması sağlamaz; halka açık veya çok kullanıcılı kullanım için tasarlanmamıştır.

## Özellikler

- PDF'den metin çıkarımı ve yapılandırılabilir parçalara bölme.
- `intfloat/multilingual-e5-small` ile yerel embedding, cosine similarity ile bellek içi semantik arama.
- Yerel Ollama modeliyle Türkçe RAG yanıtı ve sayfa/parça kaynakları.
- FastAPI API'si ve tek ekranlı Streamlit arayüzü.
- Oturum başına belge, boyut, sayfa, parça ve yaşam süresi sınırları.
- OCR yoktur: taranmış/görüntü tabanlı PDF'ler desteklenmez.

## Teknolojiler ve veri akışı

```text
Tarayıcı → Streamlit → FastAPI → pypdf → chunker → E5 embedding
                                      ↓                 ↓
                                Ollama ← RAG ← bellek içi vektör indeksi
```

Python 3.11/3.12, FastAPI, Streamlit, pypdf, sentence-transformers, Ollama, pytest ve Ruff kullanılır. Ücretli API veya harici embedding servisi kullanılmaz. Ollama uygulaması ve modelleri bu projeden ayrı kurulur.

## Windows kurulumu

PowerShell'de proje klasöründe:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[ml,web,dev]"
Copy-Item .env.example .env
hf download intfloat/multilingual-e5-small
```

İlk paket/model indirmesi internet, birkaç GB disk alanı ve çıkarım sırasında CPU/RAM kullanır. E5 modeli CPU'da çalışacak şekilde yapılandırılmıştır. Model cache'i Git'e eklenmemelidir. `EMBEDDING_MODEL` değişirse mevcut indeks o modelle üretilmediğinden yeniden oluşturulmalıdır.

Ollama'yı [resmi indirme sayfasından](https://ollama.com/download) kurup yerel sohbet modelini alın:

```powershell
ollama pull qwen3:1.7b
```

Ardından iki ayrı terminalde API ve arayüzü çalıştırın:

```powershell
# Terminal 1 — yalnızca yerel bilgisayarda dinleyin
python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```

```powershell
# Terminal 2
python -m streamlit run frontend/app.py
```

Arayüz Streamlit'in bildirdiği yerel adreste açılır. API varsayılan olarak `http://localhost:8000`, Ollama `http://127.0.0.1:11434` adresindedir. Ollama yerel uygulaması çalışmıyorsa gerekiyorsa ayrı terminalde `ollama serve` başlatın.

## Yapılandırma ve kullanım

`.env.example`, `EMBEDDING_MODEL`, `OLLAMA_MODEL`, CPU cihazı, `TOP_K`, PDF sınırları ve chunk boyutlarını listeler. Embedding ve üretim modeli birbirinden bağımsız seçilir. Varsayılan ayarlar ücretli servis kullanmaz; uzak Ollama host'u ayrıca açıkça etkinleştirilmelidir. Uzak host etkinleştirilirse soru ve seçilen belge parçaları o ağa gönderilir.

1. PDF'yi arayüzden yükleyip indeksleyin.
2. Belgedeki bir bilgi hakkında Türkçe soru sorun.
3. Yanıt altındaki kaynakları açarak dosya, sayfa, parça ve benzerlik sıralama puanını inceleyin.

Benzerlik puanı olasılık veya doğruluk yüzdesi değildir. Sohbet geçmişi yalnızca ekranda gösterilir ve sonraki isteklere bağlam olarak gönderilmez. Oturum temizleme, backend'deki geçici indeks ve belge özetlerini siler.

## Ekran görüntüleri

Bu sürümde depoya kaydedilmiş gerçek uygulama ekran görüntüsü yoktur. Portföy paylaşımından önce yerel uygulamadan masaüstü ve dar mobil genişlikte, yalnızca sentetik/kamuya açık içerik kullanarak ekran görüntüsü ekleyin. Gerçek kullanıcı belgesi veya hassas veri göstermeyin.

## Test ve kalite kontrolleri

Offline testler model indirmez ve Ollama servisi gerektirmez:

```powershell
python -m ruff format .
python -m ruff check .
python -m ruff format --check .
python -m pytest -v
python -m pip check
```

Gerçek E5 entegrasyon ve sentetik retrieval değerlendirmesini ayrıca, açıkça etkinleştirerek çalıştırın:

```powershell
$env:RUN_MODEL_INTEGRATION = "1"
try { python -m pytest -m integration -v -s }
finally { Remove-Item Env:RUN_MODEL_INTEGRATION -ErrorAction SilentlyContinue }
```

Gerçek Ollama/RAG testi ayrıca ve isteğe bağlıdır; Ollama'nın çalışmasını ve modelin yerel olarak bulunmasını ister:

```powershell
$env:RUN_OLLAMA_INTEGRATION = "1"
try { python -m pytest -m ollama_integration -v -s }
finally { Remove-Item Env:RUN_OLLAMA_INTEGRATION -ErrorAction SilentlyContinue }
```

Küçük sentetik retrieval seti genel Türkçe arama veya yanıt başarısını kanıtlamaz. Model kalitesi, kaynak tutarlılığı ve donanıma bağlı gecikme daha geniş bir değerlendirme gerektirir.

## Güvenlik ve sınırlar

- `.env`, Streamlit secrets, model cache'leri, sanal ortamlar ve build çıktıları Git'ten yok sayılır. `.env.example` içinde gerçek sır bulunmamalıdır.
- Ham PDF metni, sorular, prompt'lar ve anahtarlar loglanmaz. Yerel model indirmek için internet erişimi gerekir; çıkarım varsayılan olarak yereldir.
- Uygulama kimlik doğrulaması, kullanıcılar arası erişim kontrolü veya canlı dağıtım sağlamaz. Halka açık kullanıma açmayın ve kişisel/gizli belge yüklemeyin.
- PDF sayfa/boyut sınırları riskleri azaltır ancak parser güvenlik açığını veya kötü amaçlı dosyayı bütünüyle engellemez. Ayrıntı için [SECURITY.md](SECURITY.md) dosyasına bakın.
- RAM ve yanıt süresi seçilen embedding/LLM modeli, bağlam uzunluğu ve bilgisayar donanımına bağlıdır. Daha büyük Ollama modelleri daha fazla bellek isteyebilir.

## Ücretsiz demo ve dağıtım değerlendirmesi

Canlı dağıtım bu depoda yapılandırılmamıştır. Resmi sağlayıcı belgelerine göre mevcut seçeneklerin özeti:

| Seçenek | Ücret ve kaynak koşulları | Bu projeye uygunluk |
| --- | --- | --- |
| Streamlit Community Cloud | Ücretsiz uygulama yayınlama sunar. Dokümanlarda yaklaşık 0.078–2 CPU, 690 MB–2.7 GB RAM ve en fazla 50 GB storage belirtilir (limit sayfası yaklaşık değerleri Şubat 2024 olarak tarihlendiriyor); uygulamalar 12 saat trafiksiz kalınca uyur. Uygulamalar ABD'de barındırılır. | E5 + Ollama'nın birlikte çalışması RAM ve başlatma sınırlarına takılabilir. Daha önemlisi mevcut uygulamanın API/Ollama `localhost` adresleri bulut makinesine işaret eder; o nedenle mimari değiştirilmeden demo çalışmaz. PDF bulut servise yüklenir, bu yüzden özel belge için uygun değildir. Model cache'inin disk kalıcılığı garanti edilmez. |
| Hugging Face Spaces | CPU Basic listelenen tabloda 2 vCPU, 16 GB RAM, 50 GB geçici disk ve ücretsiz saatlik maliyet olarak geçer; disk durdurma/yeniden başlatmada silinir, varsayılan uyku 48 saattir. Yeni Gradio/Docker compute Space oluşturmak kişisel hesapta PRO planı gerektirir; ZeroGPU istisnası Gradio ile sınırlıdır. Statik Space ücretsiz olsa da Python compute çalıştırmaz. | 16 GB bellek bu uygulama için daha elverişli görünse de Ollama ve E5'yi aynı yerde çalıştırma, başlangıç süresi, yeniden indirilmesi gereken model cache'i ve servis portları tasarım gerektirir. Kullanıcı PDF'si bulut makinesinde işlenir. 0 TL garanti edilemez. |
| Yerel çalışma | Kendi bilgisayarınızın CPU/RAM/diskini kullanır; uygulama ücreti yoktur. İlk kurulumda internet ve birkaç GB model alanı gerekir. | Mevcut mimarinin çalıştığı yol budur. Belge metni varsayılan olarak yerel API ve yerel Ollama sınırında kalır. |
| GitHub | Kaynak kodu sürüm kontrolü ve portföy paylaşımı için kullanılabilir; tek başına Python API veya Streamlit sunucusu çalıştırmaz. | Kod deposu yayınlama çözümünün yerine geçmez. Bu klasörde henüz Git deposu oluşturulmamıştır. |

Kaynaklar: [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud), [Streamlit kaynak limitleri ve uyku](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app), [Streamlit barındırma bölgesi](https://docs.streamlit.io/deploy/streamlit-community-cloud/status), [Hugging Face Spaces donanım/plan koşulları](https://huggingface.co/docs/hub/en/spaces-overview), [Hugging Face Spaces geçici disk](https://huggingface.co/docs/hub/spaces-storage), [Hugging Face ücretsiz CPU uyku süresi](https://huggingface.co/docs/hub/spaces-gpus). Şartlar ve kaynak sınırları değişebilir; herhangi bir demo öncesi yeniden doğrulayın.

**Öneri:** Yerel sürümü temel ürün olarak tutun. Canlı demo için uygulama mimarisi ve belge gizliliği ayrıca tasarlanmadan ücretsiz hosting vaadinde bulunmayın.

## Proje durumu

Bu bir yerel portföy MVP'sidir; tam güvenlik denetimi yapılmamış ve üretime hazır olduğu iddia edilmemektedir. GitHub deposu oluşturma, push ve deployment kullanıcı kontrolünde kalır. Mimari ve gizlilik önerileri için [SECURITY.md](SECURITY.md) dosyasını inceleyin.
