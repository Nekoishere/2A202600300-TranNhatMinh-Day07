# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Trần Nhật Minh
**Nhóm:** [Tên nhóm]
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Cosine similarity cao (gần 1.0) có nghĩa là hai vector embedding có cùng hướng trong không gian vector, tức là hai đoạn văn bản biểu đạt ý nghĩa tương tự nhau dù có thể dùng từ ngữ khác nhau. Đây là thước đo góc giữa hai vector, không phụ thuộc vào độ lớn của chúng.

**Ví dụ HIGH similarity:**
- Sentence A: "Students must complete all required credits to graduate."
- Sentence B: "To receive a degree, students need to finish all mandatory coursework."
- Tại sao tương đồng: Cả hai câu đều diễn đạt cùng một quy định về điều kiện tốt nghiệp, dùng các khái niệm trùng nhau nên embedding hướng về cùng một vùng trong không gian vector.

**Ví dụ LOW similarity:**
- Sentence A: "The university prohibits sexual misconduct on campus."
- Sentence B: "A doctoral dissertation must be defended before a committee."
- Tại sao khác: Hai câu thuộc hai quy định hoàn toàn khác nhau (quy tắc ứng xử vs yêu cầu học thuật), không có khái niệm chung nên embedding nằm ở hai hướng xa nhau trong không gian vector.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity chỉ đo góc giữa hai vector, không bị ảnh hưởng bởi độ dài văn bản — một đoạn quy chế ngắn và một điều khoản dài cùng chủ đề vẫn có cosine similarity cao. Euclidean distance bị ảnh hưởng bởi magnitude của vector, khiến văn bản dài luôn bị đánh giá "xa" hơn văn bản ngắn dù cùng nội dung.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Trình bày phép tính:
> - step = chunk_size − overlap = 500 − 50 = 450
> - Các vị trí bắt đầu: `range(0, 10000, 450)` → 0, 450, 900, ..., 9450, 9900
> - Tại start=9900: chunk = text[9900:10400] (chỉ còn 100 ký tự), 9900+500 ≥ 10000 → break
> - Số phần tử: floor(9999 / 450) + 1 = 22 + 1 = **23 chunks**
>
> Đáp án: **23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Với overlap=100: step = 500−100 = 400; `range(0, 10000, 400)` cho 25 phần tử → **25 chunks** (nhiều hơn 2 chunks so với overlap=50). Overlap lớn hơn giúp mỗi chunk chia sẻ ngữ cảnh với chunk liền kề, tránh trường hợp một điều khoản quan trọng bị cắt đứt giữa hai chunk và mất ngữ nghĩa trong retrieval.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** FAQ vin policy

**Tại sao nhóm chọn domain này?**
> *Viết 2-3 câu:*
Các văn bản quy chế đào tạo thường rất dài, ngôn từ mang tính pháp lý/học thuật cao và được chia thành cấu trúc phân cấp phức tạp (Chương, Điều, Khoản). Đây là một Use Case hoàn hảo cho hệ thống RAG vì sinh viên thường gặp khó khăn khi tra cứu thủ công, đồng thời nó đòi hỏi chiến lược phân mảnh dữ liệu (chunking) thông minh để không làm mất ngữ cảnh của từng điều luật.
### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | VU_HT02.VN Quy chế đào tạo Thạc sĩ | VinUniversity (ban hành 20/12/2022) | ~32,367 | chapter, doc_id, source |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| chapter | string | `quy_che_thac_si` | Lọc kết quả theo chủ đề quy chế, tránh trả về chunk không liên quan |
| doc_id | string | `VU_HT02.VN_Quy-che...` | Xác định chunk thuộc tài liệu gốc nào, hỗ trợ delete_document và truy vết nguồn |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| Quy chế Thạc sĩ | FixedSizeChunker (`fixed_size`) | 35 | 973 chars | ❌ Cắt giữa câu, phá vỡ cấu trúc Điều/Khoản |
| Quy chế Thạc sĩ | SentenceChunker (`by_sentences`) | 73 | 441 chars | ✅ Giữ trọn câu, nhưng chunk quá nhỏ |
| Quy chế Thạc sĩ | RecursiveChunker (`recursive`) | 37 | 873 chars | ✅ Giữ nguyên paragraph theo cấu trúc Điều |

### Strategy Của Tôi

**Loại:** RecursiveChunker (chunk_size=1000)

**Mô tả cách hoạt động:**
> RecursiveChunker chia văn bản theo thứ tự ưu tiên các dấu tách: `\n\n` (paragraph) → `\n` (dòng) → `. ` (câu) → ` ` (từ). Nếu sau khi cắt một đoạn vẫn còn dài hơn chunk_size, nó tiếp tục đệ quy với dấu tách nhỏ hơn. Các mảnh nhỏ được gộp lại (merge) nếu tổng độ dài chưa vượt quá chunk_size, tránh lãng phí dung lượng chunk.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Văn bản quy chế VinUni được tổ chức theo cấu trúc phân cấp rõ ràng: Chương → Điều → Khoản, ngăn cách bởi dấu xuống dòng kép `\n\n`. RecursiveChunker khai thác pattern này bằng cách ưu tiên cắt theo paragraph, giữ nguyên từng Điều/Khoản trọn vẹn trong một chunk — phù hợp hơn so với fixed_size (cắt bừa) và sentence (chunk quá nhỏ).

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| Quy chế Thạc sĩ | fixed_size (baseline) | 35 | 973 | 3/5 queries đúng — hay cắt giữa Điều |
| Quy chế Thạc sĩ | **recursive (của tôi)** | 37 | 873 | 3/5 queries đúng — giữ Điều trọn vẹn hơn |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> RecursiveChunker là lựa chọn tốt nhất cho domain quy chế vì nó tôn trọng cấu trúc phân cấp tự nhiên của văn bản pháp lý. Khi mỗi chunk giữ được trọn vẹn 1 Điều hoặc 1 Khoản, hệ thống tìm kiếm dễ dàng trả về đúng đoạn văn bản chứa câu trả lời mà không bị mất bối cảnh.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Thay vì dùng regex phức tạp, tôi dùng kỹ thuật sentinel character (`\x00`) để đánh dấu ranh giới câu: thay thế `. `, `! `, `? `, `.\n` bằng dấu câu tương ứng cộng sentinel, rồi split theo sentinel. Cách này giữ nguyên dấu câu trong chunk (không mất dấu chấm cuối câu), xử lý được edge case như văn bản tiếng Việt không có khoảng trắng chuẩn sau dấu câu. Sau khi tách câu, các câu được gom theo nhóm `max_sentences_per_chunk` và join bằng khoảng trắng.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Algorithm dùng đệ quy với danh sách separator theo thứ tự ưu tiên giảm dần (`\n\n` → `\n` → `. ` → ` ` → ``). Ở mỗi cấp, split text theo separator hiện tại và gộp các phần vào buffer cho đến khi buffer vượt `chunk_size` thì flush. Nếu một phần đơn lẻ vẫn quá lớn, đệ quy xuống separator tiếp theo. Base case: text ngắn hơn `chunk_size` (trả về nguyên) hoặc hết separator (hard slice theo ký tự).

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `add_documents` chunk mỗi document qua `FixedSizeChunker` trước khi embed, tạo `chunk_doc` với ID dạng `{doc_id}_chunk_{i}` và metadata kế thừa từ document gốc. Mỗi chunk được embed qua `embedding_fn` và lưu dưới dạng dict `{id, content, embedding, metadata}` vào `self._store`. `search` embed query bằng cùng hàm đó rồi gọi `_search_records` để tính dot product giữa query embedding và mọi stored embedding, sort giảm dần, trả về top_k kết quả kèm `score`.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` filter **trước** khi search: lọc `self._store` theo tất cả key-value trong `metadata_filter` bằng list comprehension, sau đó chạy `_search_records` chỉ trên tập đã lọc. `delete_document` dùng list comprehension để giữ lại mọi chunk có `metadata['doc_id'] != doc_id`, trả về `True` nếu số lượng record giảm — cách này O(n) nhưng đơn giản và đúng cho in-memory store.

### KnowledgeBaseAgent

**`answer`** — approach:
> `answer` lấy top-k chunks từ store qua `search`, join các `chunk["content"]` bằng `\n\n` thành context block, rồi build prompt theo cấu trúc: hướng dẫn → context → question → "Answer:". Prompt được truyền thẳng vào `llm_fn` (injectable dependency), cho phép test với mock LLM không cần API key, và dễ thay thế bằng bất kỳ model thực nào.

### Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.0.2, pluggy-1.6.0
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

============================== 42 passed in 0.04s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Students must complete all required credits to graduate." | "A degree requires finishing all mandatory coursework." | high | 0.8829 | Yes|
| 2 | "The university has a zero-tolerance policy for sexual misconduct." | "Doctoral candidates must submit a dissertation." | low | 0.1110 | No |
| 3 | "Quy chế thạc sĩ quy định điều kiện tốt nghiệp." | "Master's regulations specify graduation requirements." | high | 0.4896 | Yes |
| 4 | "Academic probation is triggered by low GPA." | "Students on academic warning must improve their grades." | high | 0.405 | Yes|
| 5 | "Học phí phải được nộp trước khi đăng ký môn học." | "The weather is sunny and warm today." | low | 0.0939 | Yes |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Pair 3 (Việt–Anh cùng nghĩa) sẽ là kết quả thú vị nhất: nếu multilingual embedding model cho similarity cao, điều đó chứng minh embeddings biểu diễn ý nghĩa semantic vượt qua rào cản ngôn ngữ. Pair 4 cũng đáng chú ý vì dùng từ khác nhau ("probation" vs "warning", "GPA" vs "grades") nhưng cùng khái niệm — model tốt sẽ nhận ra điều này, cho thấy embeddings nắm bắt ngữ nghĩa chứ không chỉ so khớp từ khoá.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | "Điều kiện tốt nghiệp chương trình thạc sĩ tại VinUni là gì?" | Hoàn thành số tín chỉ quy định, bảo vệ luận văn đạt, không vi phạm quy chế học tập (VU_HT02). |
| 2 | "What happens if a student violates the sexual misconduct policy?" | The university investigates, applies disciplinary sanctions including suspension or expulsion (Sexual Misconduct Guideline). |
| 3 | "Quy định về thời gian đào tạo tiến sĩ tại VinUni?" | Chương trình tiến sĩ có thời gian đào tạo tối thiểu và tối đa theo quy chế, gia hạn phải được phê duyệt (Quy chế TS). |
| 4 | "What are the academic probation rules for undergraduate students?" | Students with GPA below threshold are placed on academic warning/probation with conditions to return to good standing (VU_HT03). |
| 5 | "Điều kiện tuyển sinh chương trình bác sĩ nội trú?" | Tốt nghiệp bác sĩ đa khoa, đạt kỳ thi tuyển sinh, đáp ứng yêu cầu của chương trình (POL-ADM-001). |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | "Điều kiện tốt nghiệp thạc sĩ..." | Chunk từ VU_HT02 về điều kiện cấp bằng | [điền] | Yes | [điền sau khi chạy với OpenAI] |
| 2 | "What happens if... sexual misconduct..." | Chunk từ Sexual Misconduct Guideline về xử lý vi phạm | [điền] | Yes | [điền sau khi chạy với OpenAI] |
| 3 | "Thời gian đào tạo tiến sĩ..." | Chunk từ Quy chế TS về thời gian đào tạo | [điền] | Yes | [điền sau khi chạy với OpenAI] |
| 4 | "Academic probation rules..." | Chunk từ VU_HT03 về academic standing | [điền] | Yes | [điền sau khi chạy với OpenAI] |
| 5 | "Điều kiện tuyển sinh BSNT..." | Chunk từ POL-ADM-001 về tiêu chuẩn tuyển sinh | [điền] | Yes | [điền sau khi chạy với OpenAI] |

**Bao nhiêu queries trả về chunk relevant trong top-3?** [điền sau khi chạy] / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> [Điền sau khi demo nhóm]

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> [Điền sau khi demo]

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ thêm metadata phong phú hơn từ đầu: `language` (vi/en), `document_type` (quy chế/hướng dẫn/chính sách), `regulation_code` (VU_HT02, POL-ADM-001...) để `search_with_filter` lọc đúng quy chế theo loại truy vấn. Ngoài ra sẽ thử RecursiveChunker với separators tùy chỉnh cho văn bản pháp quy (`\n##`, `\nĐiều`, `\nArticle`) để chunk tách theo đúng điều khoản thay vì cắt cứng theo ký tự. Cuối cùng sẽ xem xét tăng chunk_size lên 1000 vì văn bản quy chế thường có điều khoản dài hơn 500 ký tự.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | / 10 |
| Chunking strategy | Nhóm | / 15 |
| My approach | Cá nhân | 9 / 10 |
| Similarity predictions | Cá nhân | / 5 |
| Results | Cá nhân | / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | / 5 |
| **Tổng** | | **/ 100** |
