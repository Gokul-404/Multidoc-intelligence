import { DocumentItem, QueryResponse } from "../types";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
const API_KEY = import.meta.env.VITE_API_KEY || "sentinel-dev-key-change-in-production";

export class ApiService {
  private tenantId: string = "default_tenant";

  setTenant(tenant: string) {
    this.tenantId = tenant;
  }

  getTenant(): string {
    return this.tenantId;
  }

  private getHeaders(): HeadersInit {
    return {
      "X-Tenant-ID": this.tenantId,
      "X-API-Key": API_KEY,
    };
  }

  async checkHealth(): Promise<{ status: string; services: Record<string, string> }> {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (!res.ok) throw new Error("Health check failed");
      return await res.json();
    } catch {
      return {
        status: "offline",
        services: { qdrant: "disconnected", redis: "disconnected" },
      };
    }
  }

  async listDocuments(): Promise<DocumentItem[]> {
    try {
      const res = await fetch(`${API_BASE}/api/v1/documents`, {
        headers: this.getHeaders(),
      });
      if (!res.ok) throw new Error("Failed to load documents");
      const data = await res.json();
      const list = Array.isArray(data) ? data : data.documents || [];
      return list.map((doc: any) => ({
        id: doc.document_id || doc.id || `doc_${Date.now()}`,
        filename: doc.filename || "document.pdf",
        pages: doc.pages ?? 1,
        file_size_mb: doc.file_size_mb ?? (doc.file_size_bytes ? doc.file_size_bytes / (1024 * 1024) : 1.0),
        language: doc.language || "en",
        document_type: doc.document_type || "general",
        is_scanned: doc.is_scanned ?? doc.ocr_used ?? false,
        total_chunks: doc.total_chunks ?? doc.chunks ?? 0,
        status: doc.status || "ready",
        uploaded_at: doc.uploaded_at || new Date().toISOString(),
        error: doc.error_message,
      }));
    } catch {
      return [];
    }
  }

  private getFallbackDocuments(): DocumentItem[] {
    return [
      {
        id: "doc_alpha_q3",
        filename: "Alphabet_Q3_2024_Financials.pdf",
        pages: 14,
        file_size_mb: 2.4,
        language: "en",
        document_type: "financial_report",
        is_scanned: false,
        total_chunks: 48,
        status: "ready",
        uploaded_at: new Date(Date.now() - 3600000).toISOString(),
      },
      {
        id: "doc_msa_2024",
        filename: "Enterprise_Master_Services_Agreement.pdf",
        pages: 28,
        file_size_mb: 4.1,
        language: "en",
        document_type: "legal_contract",
        is_scanned: false,
        total_chunks: 92,
        status: "ready",
        uploaded_at: new Date(Date.now() - 7200000).toISOString(),
      },
      {
        id: "doc_raft_arch",
        filename: "Distributed_Consensus_Architecture.pdf",
        pages: 9,
        file_size_mb: 1.1,
        language: "en",
        document_type: "technical_spec",
        is_scanned: false,
        total_chunks: 35,
        status: "ready",
        uploaded_at: new Date(Date.now() - 14400000).toISOString(),
      },
    ];
  }

  async uploadDocument(file: File): Promise<DocumentItem> {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${API_BASE}/api/v1/documents/upload`, {
      method: "POST",
      headers: this.getHeaders(),
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Upload failed" }));
      throw new Error(err.detail || "Failed to upload document");
    }

    const data = await res.json().catch(() => ({}));
    const sizeMb = Number((file.size / (1024 * 1024)).toFixed(2));
    return {
      id: data.document_id || `doc_${Date.now()}`,
      filename: data.filename || file.name,
      pages: 1,
      file_size_mb: sizeMb,
      language: "en",
      document_type: "general",
      is_scanned: false,
      total_chunks: 1,
      status: "ready",
      uploaded_at: new Date().toISOString(),
    };
  }

  async deleteDocument(documentId: string): Promise<boolean> {
    try {
      const res = await fetch(`${API_BASE}/api/v1/documents/${documentId}`, {
        method: "DELETE",
        headers: this.getHeaders(),
      });
      return res.ok;
    } catch {
      return true;
    }
  }

  async query(
    question: string,
    documentIds: string[],
    conversationId?: string,
    onChunk?: (token: string) => void
  ): Promise<QueryResponse> {
    try {
      const res = await fetch(`${API_BASE}/api/v1/query`, {
        method: "POST",
        headers: {
          ...this.getHeaders(),
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question,
          document_ids: documentIds.length ? documentIds : undefined,
          conversation_id: conversationId,
          developer_mode: true,
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(errData.detail || `Query failed: ${res.statusText}`);
      }

      const data: QueryResponse = await res.json();
      if (onChunk && data.answer) {
        // Smooth simulated streaming for crisp visual feedback
        const words = data.answer.split(" ");
        let current = "";
        for (const w of words) {
          current += (current ? " " : "") + w;
          onChunk(current);
          await new Promise((r) => setTimeout(r, 15));
        }
      }
      return data;
    } catch (err: any) {
      console.error("Query error:", err);
      throw err;
    }
  }

  private async generateOfflineDemoResponse(question: string, docIds: string[]): Promise<QueryResponse> {
    await new Promise((r) => setTimeout(r, 800));

    const isRefusal = question.toLowerCase().includes("coca-cola") || question.toLowerCase().includes("secret recipe");

    if (isRefusal) {
      return {
        answer: "I cannot answer this question based on the provided documents. The uploaded documents contain financial reports, legal agreements, and distributed systems specifications, but contain no information regarding the queried topic.",
        citations: [],
        confidence_score: 12.0,
        grounded: false,
        refused: true,
        refusal_reason: "Zero supporting evidence chunks retrieved across selected documents.",
        retries_attempted: 1,
        audit: {
          status: "FAIL",
          confidence_score: 12.0,
          grounded: false,
          claims_total: 0,
          claims_supported: 0,
          claims_unsupported: 0,
          citations_valid: 0,
          citations_invalid: 0,
          has_numerical_claims: false,
          numerical_claims_verified: false,
          has_contradictions: false,
          failure_reasons: ["Zero relevant chunks above cosine similarity threshold (0.65). Answer refused to prevent hallucination."],
        },
      };
    }

    return {
      answer: "According to the financial documentation, consolidated revenues for Q3 2024 reached $88.3 billion [Doc: Alphabet_Q3_2024_Financials.pdf, P. 3], representing a 15% YoY increase compared to the prior year period. In addition, operating margins expanded to 32% driven by cloud infrastructure efficiencies.",
      citations: [
        {
          citation_index: 1,
          document_id: docIds[0] || "doc_alpha_q3",
          document_name: "Alphabet_Q3_2024_Financials.pdf",
          page_number: 3,
          chunk_id: "alpha_rev_chunk_03",
          supporting_text: "Consolidated revenues were $88.3 billion in the third quarter of 2024, an increase of 15% year over year.",
        },
        {
          citation_index: 2,
          document_id: docIds[0] || "doc_alpha_q3",
          document_name: "Alphabet_Q3_2024_Financials.pdf",
          page_number: 5,
          chunk_id: "alpha_margin_chunk_05",
          supporting_text: "Operating margins expanded to 32% reflecting operating leverage and disciplined headcount allocation.",
        },
      ],
      confidence_score: 96.5,
      grounded: true,
      refused: false,
      retries_attempted: 0,
      audit: {
        status: "PASS",
        confidence_score: 96.5,
        grounded: true,
        claims_total: 3,
        claims_supported: 3,
        claims_unsupported: 0,
        citations_valid: 2,
        citations_invalid: 0,
        has_numerical_claims: true,
        numerical_claims_verified: true,
        has_contradictions: false,
        failure_reasons: [],
        claims_breakdown: [
          {
            claim: "Consolidated revenues reached $88.3 billion in Q3 2024",
            supported: true,
            confidence: 0.99,
            supporting_chunk_ids: ["alpha_rev_chunk_03"],
          },
          {
            claim: "Revenue represents a 15% YoY increase",
            supported: true,
            confidence: 0.98,
            supporting_chunk_ids: ["alpha_rev_chunk_03"],
          },
          {
            claim: "Operating margins expanded to 32%",
            supported: true,
            confidence: 0.95,
            supporting_chunk_ids: ["alpha_margin_chunk_05"],
          },
        ],
      },
      developer_mode: {
        original_query: question,
        rewritten_queries: [
          question,
          "Alphabet Q3 2024 quarterly revenue growth rate YoY",
          "Consolidated quarterly revenue financial results 2024 Q3",
        ],
        sub_questions: [
          "What were the consolidated revenues for Q3 2024?",
          "What was the YoY revenue growth rate?",
        ],
        retrieval_method: "hybrid_rrf_with_rerank",
        dense_results_count: 15,
        bm25_results_count: 15,
        fused_results_count: 8,
        reranked_results_count: 4,
        raw_tokens_count: 1840,
        compressed_tokens_count: 620,
        compression_ratio: 0.663,
        retrieval_latency_ms: 142.5,
        generation_latency_ms: 380.2,
        total_latency_ms: 522.7,
        pipeline_stages: [
          { node: "query_analyzer", duration_ms: 45, status: "success" },
          { node: "query_rewriter", duration_ms: 62, status: "success" },
          { node: "hybrid_retriever", duration_ms: 88, status: "success" },
          { node: "flashrank_reranker", duration_ms: 35, status: "success" },
          { node: "parent_context_fetcher", duration_ms: 22, status: "success" },
          { node: "contextual_compressor", duration_ms: 48, status: "success" },
          { node: "grounded_generator", duration_ms: 380, status: "success" },
          { node: "evidence_auditor", duration_ms: 64, status: "success" },
        ],
        retrieved_chunks: [
          {
            chunk_id: "alpha_rev_chunk_03",
            document_name: "Alphabet_Q3_2024_Financials.pdf",
            page: 3,
            score: 0.942,
            retrieval_method: "hybrid_rrf",
            text: "Consolidated revenues were $88.3 billion in the third quarter of 2024, an increase of 15% year over year.",
          },
          {
            chunk_id: "alpha_margin_chunk_05",
            document_name: "Alphabet_Q3_2024_Financials.pdf",
            page: 5,
            score: 0.887,
            retrieval_method: "hybrid_rrf",
            text: "Operating margins expanded to 32% reflecting operating leverage and disciplined headcount allocation.",
          },
        ],
      },
    };
  }
}

export const api = new ApiService();
