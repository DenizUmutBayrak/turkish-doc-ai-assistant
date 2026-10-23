"""Small synthetic Turkish retrieval evaluation using the real local model."""

import json
import os

import pytest

from backend.core.config import Settings
from backend.models.documents import TextChunk
from backend.services.embedding_service import EmbeddingService
from backend.services.retrieval import InMemoryVectorIndex

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_MODEL_INTEGRATION") != "1",
        reason="Model indirme/çalıştırma için RUN_MODEL_INTEGRATION=1 ayarlayın.",
    ),
]


@pytest.fixture
def synthetic_chunks() -> list[TextChunk]:
    texts = [
        (
            "garanti.pdf",
            2,
            "Ürünler satın alma tarihinden itibaren 24 ay boyunca "
            "üretim hatalarına karşı garantilidir.",
        ),
        (
            "garanti.pdf",
            3,
            "Garanti başvurusu için fatura ve ürün seri numarası teknik servise iletilmelidir.",
        ),
        (
            "iade.pdf",
            1,
            "Kullanılmamış ürünler teslim tarihinden sonraki 30 gün içinde iade edilebilir.",
        ),
        (
            "iade.pdf",
            2,
            "İade talebinde ürünün kutusu, aksesuarları ve faturası eksiksiz olmalıdır.",
        ),
        (
            "destek.pdf",
            1,
            "Teknik destek ekibine hafta içi 09.00 ile 18.00 saatleri arasında "
            "telefonla ulaşabilirsiniz.",
        ),
        (
            "destek.pdf",
            2,
            "Arıza kaydı oluştururken hata mesajını ve cihazın model numarasını paylaşın.",
        ),
        (
            "kargo.pdf",
            1,
            "Stoktaki siparişler ödeme onayından sonra iki iş günü içinde kargoya verilir.",
        ),
        (
            "kargo.pdf",
            2,
            "Kargo takip bağlantısı gönderi firmaya teslim edildiğinde e-posta ile iletilir.",
        ),
        (
            "hesap.pdf",
            1,
            "Hesabınızı korumak için güçlü bir parola belirleyin ve doğrulama kodunu "
            "kimseyle paylaşmayın.",
        ),
        (
            "hesap.pdf",
            2,
            "Şüpheli oturumları hesap ayarlarından kapatabilir ve parolanızı "
            "hemen yenileyebilirsiniz.",
        ),
        (
            "uyelik.pdf",
            1,
            "Üyeliğinizi profilinizdeki Abonelik bölümünden istediğiniz zaman "
            "sonlandırabilirsiniz.",
        ),
        (
            "uyelik.pdf",
            2,
            "İptal işlemi dönem sonunda yürürlüğe girer; mevcut dönem sonuna kadar "
            "erişiminiz sürer.",
        ),
    ]
    return [
        TextChunk(text=text, filename=filename, page_number=page, chunk_index=index)
        for index, (filename, page, text) in enumerate(texts)
    ]


def test_real_e5_turkish_retrieval_metrics(synthetic_chunks: list[TextChunk]) -> None:
    cases = [
        (
            "Satın aldığım cihazın üretim kaynaklı arızası ne zamana kadar karşılanıyor?",
            "garanti.pdf",
        ),
        (
            "Ürünü beğenmezsem teslim aldıktan sonra hangi süre içinde geri gönderebilirim?",
            "iade.pdf",
        ),
        (
            "Cihazda arıza olduğunda müşteri hizmetlerine ne zaman telefon açabilirim?",
            "destek.pdf",
        ),
        (
            "Siparişim depodan çıktıktan sonra taşıyıcıya ne zaman teslim edilir?",
            "kargo.pdf",
        ),
        (
            "Hesabıma başkası girmiş olabilir, oturumu ve parolayı nasıl güvene alırım?",
            "hesap.pdf",
        ),
        ("Ücretli üyeliği durdurursam erişimim ne zaman sona erer?", "uyelik.pdf"),
    ]
    index = InMemoryVectorIndex(embedding_service=EmbeddingService(settings=Settings()))
    assert index.index_chunks(synthetic_chunks) == 12

    rows = []
    first_hits = 0
    recall_hits = 0
    top_k = 3
    for query, expected_filename in cases:
        results = index.search(query, top_k=top_k)
        assert len(results) == top_k
        ranks = [
            rank
            for rank, result in enumerate(results, start=1)
            if result.filename == expected_filename
        ]
        rank = ranks[0] if ranks else None
        first_hits += rank == 1
        recall_hits += rank is not None
        rows.append(
            {
                "query": query,
                "expected_source": expected_filename,
                "expected_rank": rank,
                "top_results": [
                    {
                        "source": result.filename,
                        "page": result.page_number,
                        "similarity": round(result.similarity, 4),
                    }
                    for result in results
                ],
            }
        )

    report = {
        "queries": len(cases),
        "top1_accuracy": round(first_hits / len(cases), 4),
        "recall_at_3": round(recall_hits / len(cases), 4),
        "results": rows,
        "note": "Küçük sentetik set; genel Türkçe retrieval başarısını kanıtlamaz.",
    }
    print("TURKISH_RETRIEVAL_EVALUATION=" + json.dumps(report, ensure_ascii=False))
